import numpy as np
import cProfile
import matplotlib.pyplot as plt
import time
from numba import jit
from sksparse.cholmod import cholesky
from scipy.sparse import spdiags, eye, kron, diags, csr_matrix, bmat, lil_matrix
from scipy.sparse.linalg import splu, eigs, spsolve, gmres, LinearOperator
from scipy.linalg import qr, lstsq
from scipy.io import loadmat, savemat
from scipy.interpolate import Akima1DInterpolator, interpn
import CPEO_utils_3d as cpeo
import stokes_solver_utils_fast as stokes
import ib_3d as amr_solve
import icosphere
import pyvista as pv

###########################
######  PARAMETERS  #######
###########################

## Grid parameters
Nx, Ny, Nz = 64, 64, 64  # 256; % number of grid points along one direction
L = 2.0 * np.pi 
x_lo = -L/2
x_hi = L/2
y_lo = x_lo
y_hi = x_hi
z_lo = x_lo
z_hi = x_hi
x = np.linspace(x_lo, x_hi, Nx) 
dx = x[1] - x[0]
y = x.copy()
dy = y[1] - y[0]
z = x.copy()
dz = z[1] - z[0]

## Fine grid    
ref_ratio = 2
nx_fine_patch = Nx * ref_ratio // 2   # = 64
ny_fine_patch = Ny * ref_ratio // 2   # = 64
nz_fine_patch = Nz * ref_ratio // 2   # = 64

# Fine patch physical extent: central 50% of the domain
x_patch_lo = x_lo + 0.25 * (x_hi - x_lo)
x_patch_hi = x_hi - 0.25 * (x_hi - x_lo) 
y_patch_lo = y_lo + 0.25 * (y_hi - y_lo)
y_patch_hi = y_hi - 0.25 * (y_hi - y_lo)
z_patch_lo = z_lo + 0.25 * (z_hi - z_lo)
z_patch_hi = z_hi - 0.25 * (z_hi - z_lo)

x_fine = np.linspace(x_patch_lo, x_patch_hi, nx_fine_patch)
y_fine = np.linspace(y_patch_lo, y_patch_hi, ny_fine_patch)
z_fine = np.linspace(z_patch_lo, z_patch_hi, nz_fine_patch)

dx_fine = x_fine[1] - x_fine[0]
dy_fine = y_fine[1] - y_fine[0]
dz_fine = z_fine[1] - z_fine[0]

Xf, Yf, Zf = np.meshgrid(x_fine, y_fine, z_fine, indexing='ij')

## Miscellaneous parameters
tol = 1e-3
beta_BC = 7.94
sigma_bc = 0.78  # 0.68
delta_layer = 5*dx  # 0.1 # 5*dx; %6*dx;
cut_coarse = 6 * 1.2 * dx # cutoff value
cut_fine = 6 * 1.2 * dx #dx_fine

# Anderson acceleration parameters
beta = 1.4
m = 100

##########################
######  GRID SETUP  ######
##########################

## Nodal grid (interior points)
xint = x[1:-1]
yint = y[1:-1]
zint = z[1:-1]
# Ny = len(yint)
# Nz = len(zint)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
Xint, Yint, Zint = np.meshgrid(xint, yint, zint, indexing='ij')

# JUST POISSON FOR NOW
# ## Staggered grid (horizontal faces for V, vertical faces for U, cell centers for P)
# N_U = Nx * (Ny + 1)
# N_V = (Nx + 1) * Ny
# N_P = (Nx + 1) * (Ny + 1)
# x_trunc = x[1:-1]    # length Nx
# y_trunc = y[1:-1]    # length Ny
# x_mid = x + dx / 2
# y_mid = y + dy / 2
# x_offset = x_mid[:-1]
# y_offset = y_mid[:-1]

# UGridX, UGridY = np.meshgrid(x_trunc, y_offset)
# VGridX, VGridY = np.meshgrid(x_offset, y_trunc)

## Immersed boundary
rad = 0.25

# Coarse

# Choose number of subdivisions so point spacing matches dx
# Edge length of icosphere ~ rad * 1.0 / nu (approximate)
nu_coarse = max(1, int(rad / (dx)))
vertices_coarse, faces_coarse = icosphere.icosphere(nu_coarse)

# vertices are already on the unit sphere, scale to radius
xib_coarse = rad * vertices_coarse[:, 0]
yib_coarse = rad * vertices_coarse[:, 1]
zib_coarse = rad * vertices_coarse[:, 2]

# Outward unit normals on a sphere are just the unit position vectors
n_x_coarse = vertices_coarse[:, 0]
n_y_coarse = vertices_coarse[:, 1]
n_z_coarse = vertices_coarse[:, 2]

Nib_coarse = len(vertices_coarse)

fig = plt.figure()
ax = fig.add_subplot(projection='3d') # or fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(xib_coarse, yib_coarse, zib_coarse)

# Set labels for clarity
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

# Choose number of subdivisions so point spacing matches dx
# Edge length of icosphere ~ rad * 1.0 / nu (approximate)
nu_fine = max(1, int(rad / (dx))) #dx_fine 
vertices_fine, faces_fine = icosphere.icosphere(nu_fine)

# vertices are already on the unit sphere, scale to radius
xib_fine = rad * vertices_fine[:, 0]
yib_fine = rad * vertices_fine[:, 1]
zib_fine = rad * vertices_fine[:, 2]

# Outward unit normals on a sphere are just the unit position vectors
n_x_fine = vertices_fine[:, 0]
n_y_fine = vertices_fine[:, 1]
n_z_fine = vertices_fine[:, 2]

Nib_fine = len(vertices_fine)

fig = plt.figure()
ax = fig.add_subplot(projection='3d') # or fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(xib_fine, yib_fine, zib_fine)

# Set labels for clarity
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

#########################
######  OPERATORS  ######
#########################

## Laplacian operators
# Staggered laplacians (dirichlet boundary conditions in y, dirichlet boundary conditions in x)
# Lap_U, Lap_V = stokes.build_staggered_Laps(Nx, dx)

# Nodal laplacian (dirichlet boundary conditions in y, periodic in x)
# e = (1/dy**2) * np.ones(Ny)
# D2_d = spdiags([e, -2*e, e], [-1, 0, 1], Ny, Ny)
# I_nx = eye(Nx)
# I_ny = eye(Ny)
# Lap = -(kron(I_nx, D2_d) + kron(D2_d, I_ny))
# dLap = cholesky(Lap) # Cholesky decomposition

# 1D Laplacian for Dirichlet x
e_x = (1/dx**2) * np.ones(Nx)
D2_x = spdiags([e_x, -2*e_x, e_x], [-1, 0, 1], Nx, Nx).toarray()

# 1D Laplacian for Dirichlet y
e_y = (1/dy**2) * np.ones(Ny)
D2_y = spdiags([e_y, -2*e_y, e_y], [-1, 0, 1], Ny, Ny).toarray()

# 1D Laplacian for Dirichlet z
e_z = (1/dz**2) * np.ones(Nz)
D2_z = spdiags([e_z, -2*e_z, e_z], [-1, 0, 1], Nz, Nz).toarray()

I_nx = eye(Nx)
I_ny = eye(Ny)
I_nz = eye(Nz)

# 3D Laplacian with Dirichlet conditions
Lap =  (kron(I_nz, kron(D2_x, I_ny)) +
        kron(I_nz, kron(I_nx, D2_y)) +
        kron(D2_z, kron(I_nx, I_ny)))

# # Full gradient operators ((Nx + 2) * (Ny + 2)) including boundaries
# D_x_full = diags(
#     [-0.5/dx * np.ones(Nx), 0.5/dx * np.ones(Nx)],
#     offsets=[-1, 1],
#     shape=(Nx, Nx + 2)
# ).tocsr()
# D_y_full = diags(
#     [-0.5/dy * np.ones(Ny),  0.5/dy * np.ones(Ny)],
#     offsets=[-1,  1],
#     shape=(Ny, Ny + 2),
#     format='csr'
# )
# D_z_full = diags(
#     [-0.5/dz * np.ones(Nz),  0.5/dy * np.ones(Nz)],
#     offsets=[-1,  1],
#     shape=(Nz, Nz + 2),
#     format='csr'
# )
# G_x_full = kron(kron(D_x_full, eye(Ny + 2, format='csr'), format='csr'), eye(Nz + 2, format='csr'))
# G_y_full = kron(kron(eye(Nx + 2, format='csr'), D_y_full, format='csr'), eye(Nz + 2, format='csr'))
# G_z_full = kron(eye(Nx + 2, format='csr'), kron(eye(Nz + 2, format='csr'), D_z_full, format='csr'))

# # Nodal gradient operators (internal points only, excludes boundaries)
# D_x_nodes = diags(
#     [-0.5/dx * np.ones(Nx), 0.5/dx * np.ones(Nx)],
#     offsets=[-1, 1],
#     shape=(Nx, Nx + 2),
#     format='csr'
# )

# D_y_nodes = diags(
#     [-0.5/dy * np.ones(Ny), 0.5/dy * np.ones(Ny)],
#     offsets=[-1,  1],
#     shape=(Ny, Ny + 2),
#     format='csr'
# )
# S_y = eye(Ny + 2, format='csr')[1:-1, :]   # (Ny) × (Ny+2)
# S_x = eye(Nx + 2, format='csr')[1:-1, :]   # (Nx) × (Nx+2)
# G_x_nodes = kron(D_x_nodes, S_y, format='csr')     # (Nx*Ny) × ((Nx+2)*(Ny+2))
# G_y_nodes = kron(S_x, D_y_nodes, format='csr')     # (Nx*Ny) × ((Nx+2)*(Ny+2))

# # Staggered gradient and divergence operators
# G_x_staggered, G_y_staggered, D_x_staggered, D_y_staggered = stokes.build_staggered_Grads_Divs(Nx, dx)

# ## Prefactor big Stokes operator
# Z_UV = csr_matrix((N_U, N_V))
# Z_VU = csr_matrix((N_V, N_U))
# Z_PP = csr_matrix((N_P, N_P))

# # Saddle point system
# big_L = bmat([
#     [Lap_U, Z_VU,  -G_x_staggered],
#     [Z_UV,  Lap_V, -G_y_staggered],
#     [D_x_staggered,   D_y_staggered,   Z_PP] 
# ], format='csr')

# stokes_LU = splu(big_L)

#############################################
#######  BOUNDARY/INITIAL CONDITIONS  #######
#############################################

# ## Exact solutions
# def Phi_exact(x, y, z):
#     return beta_BC * z + 0 * x + 0 * y
# def Npm_exact(x, y, z):
#     return 0 * x + 1.0

## Exact solutions
def Phi_exact(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    return np.sin(r)
def Np_exact(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    return np.exp(-np.sin(r))
def Nm_exact(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    return np.exp(np.sin(r))

Phi_initial = Phi_exact(X, Y, Z)
Np_initial = Np_exact(X, Y, Z)
Nm_initial = Nm_exact(X, Y, Z)

# # Create a structured grid
# grid = pv.StructuredGrid(X, Y, Z)

# # Add the scalar fields
# grid['Phi_exact'] = Phi_initial.ravel(order='F')
# grid['Npm_exact'] = Npm_initial.ravel(order='F')

# # Plot Phi_exact
# plotter = pv.Plotter()
# plotter.add_mesh(grid, scalars='Phi_exact', show_scalar_bar=True)
# plotter.show(title='Phi_exact')

# # Plot Npm_exact
# # Get the scalar values
# scalars = grid['Npm_exact']

# # Create opacity array: transparent if |value - 1| < 0.001, else opaque
# opacity = np.where(np.abs(scalars - 1) < 0.001, 0.0, 1.0)

# plotter = pv.Plotter()
# plotter.add_mesh(grid, scalars='Npm_exact', opacity=opacity, show_scalar_bar=True)
# plotter.show(title='Npm_exact')

# Compute exact solutions
# Phi_BC = Phi_exact(X, Y, Z) #np.zeros_like(X) + 5
# N_pm_BC = Npm_exact(X, Y, Z) #np.zeros_like(X) + 5
Phi_guess = Phi_exact(X, Y, Z) #np.zeros_like(X) + 5
N_p_guess = Np_exact(X, Y, Z) #np.zeros_like(X) + 5
N_m_guess = Nm_exact(X, Y, Z) #np.zeros_like(X) + 5
Phi_guess_fine = Phi_exact(Xf, Yf, Zf) #np.zeros_like(X) + 5
N_p_guess_fine = Np_exact(Xf, Yf, Zf) #np.zeros_like(X) + 5
N_m_guess_fine = Nm_exact(Xf, Yf, Zf) #np.zeros_like(X) + 5

## Boundary conditions for Rphi = rho system
Phi_BCs = np.zeros_like(X)
Np_BCs = np.zeros_like(X)
Nm_BCs = np.zeros_like(X)

## Dirichlet boundaries in the far field 
# z = z_lo and z = z_hi (bottom and top faces)
Phi_BCs[0, :, :] = Phi_exact(X[0, :, :], Y[0, :, :], Z[0, :, :])  # z = z_lo
Phi_BCs[-1, :, :] = Phi_exact(X[-1, :, :], Y[-1, :, :], Z[-1, :, :])  # z = z_hi

# x = x_lo and x = x_hi (left and right faces)
Phi_BCs[:, :, 0] = Phi_exact(X[:, :, 0], Y[:, :, 0], Z[:, :, 0])  # x = x_lo
Phi_BCs[:, :, -1] = Phi_exact(X[:, :, -1], Y[:, :, -1], Z[:, :, -1])  # x = x_hi

# y = y_lo and y = y_hi (front and back faces)
Phi_BCs[:, 0, :] = Phi_exact(X[:, 0, :], Y[:, 0, :], Z[:, 0, :])  # y = y_lo
Phi_BCs[:, -1, :] = Phi_exact(X[:, -1, :], Y[:, -1, :], Z[:, -1, :])  # y = y_hi

# z = z_lo and z = z_hi (bottom and top faces)
Np_BCs[0, :, :] = Np_exact(X[0, :, :], Y[0, :, :], Z[0, :, :])  # z = z_lo
Np_BCs[-1, :, :] = Np_exact(X[-1, :, :], Y[-1, :, :], Z[-1, :, :])  # z = z_hi

# x = x_lo and x = x_hi (left and right faces)
Np_BCs[:, :, 0] = Np_exact(X[:, :, 0], Y[:, :, 0], Z[:, :, 0])  # x = x_lo
Np_BCs[:, :, -1] = Np_exact(X[:, :, -1], Y[:, :, -1], Z[:, :, -1])  # x = x_hi

# y = y_lo and y = y_hi (front and back faces)
Np_BCs[:, 0, :] = Np_exact(X[:, 0, :], Y[:, 0, :], Z[:, 0, :])  # y = y_lo
Np_BCs[:, -1, :] = Np_exact(X[:, -1, :], Y[:, -1, :], Z[:, -1, :])  # y = y_hi

# z = z_lo and z = z_hi (bottom and top faces)
Nm_BCs[0, :, :] = Nm_exact(X[0, :, :], Y[0, :, :], Z[0, :, :])  # z = z_lo
Nm_BCs[-1, :, :] = Nm_exact(X[-1, :, :], Y[-1, :, :], Z[-1, :, :])  # z = z_hi

# x = x_lo and x = x_hi (left and right faces)
Nm_BCs[:, :, 0] = Nm_exact(X[:, :, 0], Y[:, :, 0], Z[:, :, 0])  # x = x_lo
Nm_BCs[:, :, -1] = Nm_exact(X[:, :, -1], Y[:, :, -1], Z[:, :, -1])  # x = x_hi

# y = y_lo and y = y_hi (front and back faces)
Nm_BCs[:, 0, :] = Nm_exact(X[:, 0, :], Y[:, 0, :], Z[:, 0, :])  # y = y_lo
Nm_BCs[:, -1, :] = Nm_exact(X[:, -1, :], Y[:, -1, :], Z[:, -1, :])  # y = y_hi

# Boundary conditions context for Schur solve of Rphi = rho 
ctxt_BCs_Schur = np.concatenate([
    Phi_BCs.ravel(order='F'),
    Np_BCs.ravel(order='F'),
    Nm_BCs.ravel(order='F'),
    np.zeros(Nib_coarse) - (sigma_bc),
    np.zeros(Nib_coarse),
    np.zeros(Nib_coarse)
])

# Boundary conditions context for full Rphi = rho system
ctxt_BCs = np.concatenate([
    Phi_BCs.ravel(order='F'),
    Np_BCs.ravel(order='F'),
    Nm_BCs.ravel(order='F'),
    np.zeros(Nib_coarse) - (sigma_bc/delta_layer),
    np.zeros(Nib_coarse),
    np.zeros(Nib_coarse)
])

# ## Initial conditions for Rphi = rho system
# ld = loadmat('BC_run_N_300_r0p25.mat')
# METHOD = 'cubic'  # equivalent to 'makima' in MATLAB

# Ny_ld = int(ld['Ny'][0, 0])
# Nx_ld = int(ld['Nx'][0, 0])
# Nib_ld = int(ld['Nib'][0, 0])
# sz = Ny_ld * Nx_ld

# ctxt_ld = ld['ctxt'].ravel(order='F')
# Phi_ld = ctxt_ld[:sz].reshape(Ny_ld, Nx_ld, order='F')
# N_p_ld = ctxt_ld[sz:2*sz].reshape(Ny_ld, Nx_ld, order='F')
# N_m_ld = ctxt_ld[2*sz:3*sz].reshape(Ny_ld, Nx_ld, order='F')
# Q_ld = ctxt_ld[3*sz:3*sz+Nib_ld]
# Q_p_ld = ctxt_ld[3*sz+Nib_ld:3*sz+2*Nib_ld]
# Q_m_ld = ctxt_ld[3*sz+2*Nib_ld:3*sz+3*Nib_ld]

# Xint_ld = ld['Xint']
# Yint_ld = ld['Yint']
# theta_ld = ld['theta'].ravel()

# # Extract the coordinate vectors from the loaded grid
# x_ld = Xint_ld[0, :]  # First row gives x-coordinates
# y_ld = Yint_ld[:, 0]  # First column gives y-coordinates

# # Interpolate initial guesses
# Phi_init = interpn((x_ld, y_ld), Phi_ld, (Xint.T, Yint.T), method='linear', bounds_error=False, fill_value=None)
# N_p_init = interpn((x_ld, y_ld), N_p_ld, (Xint.T, Yint.T), method='nearest', bounds_error=False, fill_value=None)
# N_m_init = interpn((x_ld, y_ld), N_m_ld, (Xint.T, Yint.T), method='nearest', bounds_error=False, fill_value=None)
# Q_init = Akima1DInterpolator(theta_ld, Q_ld, method="makima", extrapolate=True)(theta)
# Q_p_init = Akima1DInterpolator(theta_ld, Q_p_ld, method="makima", extrapolate=True)(theta)
# Q_m_init = Akima1DInterpolator(theta_ld, Q_m_ld, method="makima", extrapolate=True)(theta)

####################
## EXACT SOLUTION ##
####################
r_coarse = np.sqrt(X**2 + Y**2 + Z**2)
r_fine = np.sqrt(Xf**2 + Yf**2 + Zf**2)
phi_true_coarse = np.sin(r_coarse)
n_p_true_coarse = np.exp(-np.sin(r_coarse))
n_m_true_coarse = np.exp(np.sin(r_coarse))
phi_true_fine = np.sin(r_fine)
n_p_true_fine = np.exp(-np.sin(r_fine))
n_m_true_fine = np.exp(np.sin(r_fine))

ctxt_true_coarse = np.concatenate([
    phi_true_coarse.ravel(order='F'),
    n_p_true_coarse.ravel(order='F'),
    n_m_true_coarse.ravel(order='F'),
    np.full_like(xib_coarse, np.nan),
    np.full_like(xib_coarse, np.nan),
    np.full_like(xib_coarse, np.nan)
])

ctxt_true_fine = np.concatenate([
    phi_true_fine.ravel(order='F'),
    n_p_true_fine.ravel(order='F'),
    n_m_true_fine.ravel(order='F'),
    np.full_like(xib_fine, np.nan),
    np.full_like(xib_fine, np.nan),
    np.full_like(xib_fine, np.nan)
])

ctxt_coarse = np.concatenate([
    Phi_guess.ravel(order='F'),
    N_p_guess.ravel(order='F'),
    N_m_guess.ravel(order='F'),
    np.zeros(Nib_coarse),
    np.zeros(Nib_coarse),
    np.zeros(Nib_coarse)
])

ctxt_fine = np.concatenate([
    Phi_guess_fine.ravel(order='F'),
    N_p_guess_fine.ravel(order='F'),
    N_m_guess_fine.ravel(order='F'),
    np.zeros(Nib_fine),
    np.zeros(Nib_fine),
    np.zeros(Nib_fine)
])

guess = np.zeros(Nib_coarse * 3 + Nib_fine * 3)

# Extract solution components
index_coarse = Nx * Ny * Nz
Phi_coarse = ctxt_coarse[:index_coarse].reshape((Nz, Ny, Nx), order='F')
Np_coarse = ctxt_coarse[index_coarse:2*index_coarse].reshape((Nz, Ny, Nx), order='F')
Nm_coarse = ctxt_coarse[2*index_coarse:3*index_coarse].reshape((Nz, Ny, Nx), order='F')

index_fine = nx_fine_patch * ny_fine_patch * nz_fine_patch
Phi_fine = ctxt_fine[:index_fine].reshape((nz_fine_patch, ny_fine_patch, nx_fine_patch), order='F')
Np_fine = ctxt_fine[index_fine:2*index_fine].reshape((nz_fine_patch, ny_fine_patch, nx_fine_patch), order='F')
Nm_fine = ctxt_fine[2*index_fine:3*index_fine].reshape((nz_fine_patch, ny_fine_patch, nx_fine_patch), order='F')

# ## Boundary conditions for Nu = F system
# f_bc_mat = np.zeros((Ny + 1, Nx))
# g_bc_mat = np.zeros((Ny, Nx + 1))
# h_bc = np.zeros((Ny + 1, Nx + 1)).ravel(order='F')
# z_x = np.zeros(Nib)
# z_y = np.zeros(Nib)

# ## Initial conditions for Nu = F system
# U_fluid = np.zeros(Nx * Ny)
# V_fluid = np.zeros(Nx * Ny)

###############################
#######  SETUP METHODS  #######
###############################

@jit(nopython=True)
def delta_a_3d(r, a):
    return (1/(2*np.pi*a**2)**(1.5)) * np.exp(-0.5*(r/a)**2)

@jit(nopython=True)
def delta_coarse(r):
    return delta_a_3d(r, 1.2*dx)

@jit(nopython=True)
def delta_r_coarse(r):
    # True d(delta_a)/dr = -(r/a^2) * delta_a  -- strictly negative
    return -(1/(1.2*dx)**2) * r * delta_a_3d(r, 1.2*dx)

@jit(nopython=True)
def delta_fine(r):
    return delta_a_3d(r, 1.2*dx) #dx_fine

@jit(nopython=True)
def delta_r_fine(r):
    return -(1/(1.2*dx)**2) * r * delta_a_3d(r, 1.2*dx) #dx_fine

def Sop_prime(q):
    return cpeo.spreadQ_prime_3d(X, Y, Z, xib_coarse, yib_coarse, zib_coarse, n_x_coarse, n_y_coarse, n_z_coarse, q, delta_r_coarse, cut_coarse, dx, dy, dz)

def Sop_prime_fine(q):
    return cpeo.spreadQ_prime_3d(Xf, Yf, Zf, xib_fine, yib_fine, zib_fine, n_x_fine, n_y_fine, n_z_fine, q, delta_r_fine, cut_fine, dx_fine, dy_fine, dz_fine) #dx_fine, dy_fine, dz_fine

def Jop(P):
    return cpeo.interpPhi_3d(X, Y, Z, xib_coarse, yib_coarse, zib_coarse, P, delta_coarse, cut_coarse, dx, dy, dz)

def Jop_fine(P):
    return cpeo.interpPhi_3d(Xf, Yf, Zf, xib_fine, yib_fine, zib_fine, P, delta_fine, cut_fine, dx_fine, dy_fine, dz_fine) #dx_fine, dy_fine, dz_fine

def Jop_prime(P):
    return cpeo.interpPhi_prime_3d(X, Y, Z, xib_coarse, yib_coarse, zib_coarse, n_x_coarse, n_y_coarse, n_z_coarse, P, delta_r_coarse, cut_coarse, dx, dy, dz)

def Jop_prime_fine(P):
    return cpeo.interpPhi_prime_3d(Xf, Yf, Zf, xib_fine, yib_fine, zib_fine, n_x_fine, n_y_fine, n_z_fine, P, delta_r_fine, cut_fine, dx_fine, dy_fine, dz_fine) #dx_fine, dy_fine, dz_fine

def G_d_G_3d(Phi, N_pm):
    return cpeo.Grad_dot_Grad_3d(Phi, N_pm, dx, dy, dz, Nx, Ny, Nz)

def G_d_G_3d_fine(Phi, N_pm):
    return cpeo.Grad_dot_Grad_3d(Phi, N_pm, dx_fine, dy_fine, dz_fine, nx_fine_patch, ny_fine_patch, nz_fine_patch)

# def G_d_G(Phi, N_pm):
#     return cpeo.Grad_dot_Grad(Phi, N_pm, dx, dy, Nx, Ny, Phi_BC, N_pm_BC)

def b_Op_3d(ctxt):
    print(f"b_Op_3d input norm: {np.linalg.norm(ctxt)}")
    return cpeo.Build_RHS_3d(ctxt, ctxt_BCs, Lap, G_d_G_3d, delta_layer, Nx, Ny, Nz, Nib_coarse, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, Jop, Jop_prime)

def b_Op_Schur_3d_double_grid(ctxt_coarse, ctxt_fine):
    return cpeo.Build_RHS_Schur_System_3d_double_grid(ctxt_coarse, ctxt_fine, ctxt_BCs_Schur, G_d_G_3d, G_d_G_3d_fine, delta_layer, Nx, Ny, Nz, nx_fine_patch, ny_fine_patch, nz_fine_patch, Nib_coarse, Nib_fine, Jop, Jop_fine, Jop_prime, Jop_prime_fine, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)

def b_Op_Schur_3d(ctxt):
    return cpeo.Build_RHS_Schur_System_3d(ctxt, ctxt_BCs_Schur, G_d_G_3d, delta_layer, Nx, Ny, Nz, Nib_coarse, Jop, Jop_prime, Sop_prime, dx, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)

def b_Op_MMS_coarse():
    return cpeo.Build_RHS_Schur_System_Manufactured_Solution_3d(ctxt_true_coarse, X, Y, Z, xib_coarse, yib_coarse, zib_coarse, Nx, Ny, Nz, Nib_coarse, delta_layer)

def b_Op_MMS_fine():
    return cpeo.Build_RHS_Schur_System_Manufactured_Solution_3d(ctxt_true_fine, Xf, Yf, Zf, xib_fine, yib_fine, zib_fine, nx_fine_patch, ny_fine_patch, nz_fine_patch, Nib_fine, delta_layer)

def AxOp_3d(ctxt):
    return cpeo.Constrained_Lap_3d(ctxt, ctxt_BCs, delta_layer, Nx, Ny, Nz, Nib_coarse, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, Sop_prime, Jop_prime)

def AxLinOp_3d(shape):
    def mv(vec):
        return AxOp_3d(vec)
        
    return LinearOperator((shape, shape), matvec=mv)

# def AxOp(ctxt):
#     return cpeo.Constrained_Lap(ctxt, ctxt, dLap, delta_layer, Nx, Ny, Nib, Sop_prime, Jop_prime)

# delta_x, delta_y = stokes.make_composite_deltas(dx, n=3)

# CHECK OPERATORS FOR 3D
# RHS = b_Op_3d(ctxt)
# LHS = AxOp_3d(ctxt)

## LINEARITY TEST 
phi_test = 20 + np.sin(X) * np.sin(Y) * np.sin(Z)
phi_test_fine = 20 + np.sin(Xf) * np.sin(Yf) * np.sin(Zf)
n_test = np.sin(X) * np.sin(Y) * np.sin(Z)
n_test_fine = np.sin(Xf) * np.sin(Yf) * np.sin(Zf)
f = -2 * np.sin(X) * np.sin(Y) * np.sin(Z)
f_fine = -2 * np.sin(Xf) * np.sin(Yf) * np.sin(Zf)
bcs = 20 + np.zeros_like(X)
bcs[1:-1,1:-1,1:-1] = np.zeros_like(Xint)
zero_bcs = np.zeros_like(X)

#\nabla^2 phi + n = f
#\implies phi + \nabla^{-2} n &= \nabla^{-2} f

# quick check for manufactured solution
lap_phi_coarse = -3.0 * np.sin(X) * np.sin(Y) * np.sin(Z)
lap_phi_fine = -3.0 * np.sin(Xf) * np.sin(Yf) * np.sin(Zf)
solve_coarse_1, solve_fine_1 = amr_solve.solve_poisson_double_grid(f, f_fine, bcs, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
solve_coarse_2, solve_fine_2 =  amr_solve.solve_poisson_double_grid(n_test, n_test_fine, zero_bcs, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
phi_computed_coarse = solve_coarse_1 - solve_coarse_2
phi_computed_fine = solve_fine_1 - solve_fine_2

residual_lin_coarse = np.linalg.norm(phi_computed_coarse - phi_test) / np.linalg.norm(phi_test)
print(f'Linearity test coarse residual: {residual_lin_coarse}')
residual_lin_coarse = np.linalg.norm(phi_computed_fine - phi_test_fine) / np.linalg.norm(phi_test_fine)
print(f'Linearity test fine residual: {residual_lin_coarse}')

## DO/UNDO LAP 
lap_phi_num_coarse, lap_phi_num_fine = amr_solve.apply_poisson(phi_test, phi_test_fine, bcs, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
phi_from_lap_coarse, phi_from_lap_fine = amr_solve.solve_poisson_double_grid(lap_phi_num_coarse, lap_phi_num_fine, bcs, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)

residual_phi_coarse = np.linalg.norm(phi_from_lap_coarse - phi_test) / np.linalg.norm(phi_test)
print(f'Do/undo residual coarse: {residual_phi_coarse}')

residual_phi_fine = np.linalg.norm(phi_from_lap_fine - phi_test_fine) / np.linalg.norm(phi_test_fine)
print(f'Do/undo residual fine: {residual_phi_fine}')

## G DOT G
phi_test = 5 + np.sin(X) * np.sin(Y) * np.sin(Z)
phi_test_fine = 5 + np.sin(Xf) * np.sin(Yf) * np.sin(Zf)
n_p_test = 5 + np.sin(X) * np.sin(Y) * np.sin(Z)
n_p_test_fine = 5 + np.sin(Xf) * np.sin(Yf) * np.sin(Zf)

# Analytical solution
gdg = (np.cos(X)**2 * np.sin(Y)**2 * np.sin(Z)**2 +
       np.sin(X)**2 * np.cos(Y)**2 * np.sin(Z)**2 +
       np.sin(X)**2 * np.sin(Y)**2 * np.cos(Z)**2).ravel(order='F')
gdg_fine = (np.cos(Xf)**2 * np.sin(Yf)**2 * np.sin(Zf)**2 +
       np.sin(Xf)**2 * np.cos(Yf)**2 * np.sin(Zf)**2 +
       np.sin(Xf)**2 * np.sin(Yf)**2 * np.cos(Zf)**2).ravel(order='F')

# Numerical solution
gdg_num = G_d_G_3d(phi_test.ravel(order='F'), n_p_test.ravel(order='F'))
gdg_num_fine = G_d_G_3d_fine(phi_test_fine.ravel(order='F'), n_p_test_fine.ravel(order='F'))

# Residual
residual_coarse = np.linalg.norm(gdg_num - gdg) / np.linalg.norm(gdg)
print(f'Residual coarse: {residual_coarse}')
residual_fine = np.linalg.norm(gdg_num_fine - gdg_fine) / np.linalg.norm(gdg_fine)
print(f'Residual fine: {residual_fine}')

# ## Test basic system 
# RHS = b_Op_3d(ctxt)
# shape = 3 * Nx * Ny * Nz + 3 * Nib
# AxOp = AxLinOp_3d(shape)

# shape = 3 * Nib

# # compare initial residual
# LHS = AxOp(ctxt)
# err_curr = np.linalg.norm(LHS - RHS) / np.linalg.norm(RHS)
# print(f'Initial residual: {err_curr}')

# G_u_n, _ = gmres(AxOp, RHS, rtol=tol, restart=500, callback=lambda rk: print(f"GMRES residual: {np.linalg.norm(rk)}"))

# # initialize objects for anderson
# DU = np.full((len(RHS), m), np.nan)
# DG = np.full((len(RHS), m), np.nan)
# u_n = ctxt.copy()
# u_next = G_u_n.copy()
# G_u_next = G_u_n.copy()
# err = []

# # Anderson acceleration loop
# for inner_its in range(100000):
#     RHS = b_Op_3d(u_next)
#     G_u_next, _ = gmres(AxOp, RHS, rtol=tol, restart=500, callback=lambda rk: print(f"GMRES residual: {np.linalg.norm(rk)}"))

#     m_n = min(m, inner_its + 1)
    
#     # Store differences
#     if inner_its < m:
#         DU[:, inner_its] = u_next - u_n
#         DG[:, inner_its] = G_u_next - G_u_n
#     else:
#         DU = np.roll(DU, -1, axis=1)
#         DG = np.roll(DG, -1, axis=1)
#         DU[:, -1] = u_next - u_n
#         DG[:, -1] = G_u_next - G_u_n
    
#     f_n = G_u_next - u_next
#     DF = DG[:, :m_n] - DU[:, :m_n]
#     print(f"Norm of f_n: {np.linalg.norm(f_n)}")
    
#     # QR decomposition
#     gamma, residuals, rank, s = lstsq(DF, f_n)
#     print(f"Norm of gamma: {np.linalg.norm(gamma)}")
    
#     u_n = u_next.copy()
#     G_u_n = G_u_next.copy()
    
#     u_next = (G_u_next - DG[:, :m_n] @ gamma) - (1-beta) * (f_n - DF @ gamma)
#     print(f"Change in u_next: {np.linalg.norm(u_next - u_n)}")
    
#     # Extract solution components
#     index = Nx * Ny * Nz
#     Phi = u_next[:index].reshape((Nz, Ny, Nx), order='F')
#     Np = u_next[index:2*index].reshape((Nz, Ny, Nx), order='F')
#     Nm = u_next[2*index:3*index].reshape((Nz, Ny, Nx), order='F')
    
#     # Check convergence
#     LHS = AxOp_3d(u_next)
#     err_curr = np.linalg.norm(LHS - RHS) / np.linalg.norm(RHS)
#     print(f'Residual: {err_curr}')
#     err.append(err_curr)
    
#     print(f'Iteration {inner_its}: residual = {err_curr}')
    
#     if err_curr < 1e-4:
#         print('Rphi = rho Converged!')
#         break

# # Create a structured grid
# grid = pv.StructuredGrid(X, Y, Z)

# # Add the scalar fields
# grid['Phi_exact'] = Phi
# grid['Np_exact'] = Np
# grid['Nm_exact'] = Nm

# # Plot Phi_exact
# plotter = pv.Plotter()
# plotter.add_mesh(grid, scalars='Phi_exact', show_scalar_bar=True)
# plotter.show(title='Phi_exact')

# # Plot Npm_exact
# plotter = pv.Plotter()
# # Get the scalar values
# scalars = grid['Np_exact']

# # Create opacity array: transparent if |value - 1| < 0.001, else opaque
# opacity = np.where(np.abs(scalars - 1) < 0.001, 0.0, 1.0)

# plotter = pv.Plotter()
# plotter.add_mesh(grid, scalars='Np_exact', opacity=opacity, show_scalar_bar=True)
# plotter.show(title='Np_exact')

# # Plot Npm_exact
# plotter = pv.Plotter()
# # Get the scalar values
# scalars = grid['Nm_exact']

# # Create opacity array: transparent if |value - 1| < 0.001, else opaque
# opacity = np.where(np.abs(scalars - 1) < 0.001, 0.0, 1.0)

# plotter = pv.Plotter()
# plotter.add_mesh(grid, scalars='Nm_exact', opacity=opacity, show_scalar_bar=True)
# plotter.show(title='Nm_exact')

##########################
#####  SOLVER SETUP  #####
##########################

## Initialize variables for Rphi = rho solve
# schurRHS_coarse, schurRHS_fine = b_Op_Schur_3d_double_grid(ctxt_coarse, ctxt_fine)
schurRHS_coarse = b_Op_MMS_coarse()
schurRHS_fine = b_Op_MMS_fine()
# schurRHS_full = np.concatenate([schurRHS_coarse, schurRHS_fine])
# DU = np.full((len(schurRHS_full), m), np.nan)
# DG = np.full((len(schurRHS_full), m), np.nan)

# # some fuckery here
# mask_coarse = np.ones(X.shape, dtype=bool) # Create mask of True
# mask_coarse[int(0.25*Nx):int(0.75*Nx), int(0.25*Ny):int(0.75*Ny), int(0.25*Nz):int(0.75*Nz)] = False 

# masked_setup_coarse = X[mask_coarse]
# masked_setup_coarse = masked_setup_coarse.ravel(order='F')
# masked_setup_coarse_x3 = np.concatenate([masked_setup_coarse, masked_setup_coarse, masked_setup_coarse])

DU_coarse = np.full((len(schurRHS_coarse), m), np.nan)
DG_coarse = np.full((len(schurRHS_coarse), m), np.nan)
# DU_coarse_masked = np.full((len(masked_setup_coarse_x3), m), np.nan)
# DG_coarse_masked = np.full((len(masked_setup_coarse_x3), m), np.nan)

DU_fine = np.full((len(schurRHS_fine), m), np.nan)
DG_fine = np.full((len(schurRHS_fine), m), np.nan)

u_n_coarse = ctxt_coarse.copy()
u_n_fine = ctxt_fine.copy()
# u_n = np.concatenate([u_n_coarse, u_n_fine])

# # Define Schur operator for GMRES (IN PROG)
# shape = 3 * Nib_coarse + 3 * Nib_fine
# schurOp = cpeo.SchurLinearOperator_R_3d_double_grid(shape, ctxt_BCs_Schur, Nib_coarse, Nib_fine, delta_layer, Sop_prime, Sop_prime_fine, Jop_prime, Jop_prime_fine, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, Nx, Ny, Nz, index_coarse)
# computedRHS_coarse, computedRHS_fine = cpeo.schur_rhs_R_3d_double_grid(b_Op_MMS_coarse(), b_Op_MMS_fine(), Nx, Ny, Nz, nx_fine_patch, ny_fine_patch, nz_fine_patch, Nib_coarse, Nib_fine, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, delta_layer, Jop_prime, Jop_prime_fine)
# RHS = np.concatenate([computedRHS_coarse, computedRHS_fine])

# check, _ = gmres(schurOp, RHS, x0=guess, rtol=tol, restart=500, callback=lambda rk: print(f"GMRES residual: {np.linalg.norm(rk)}"))
# check_coarse = check[:Nib_coarse*3]
# check_fine = check[Nib_coarse*3:]
# check_coarse_processed, check_fine_processed = cpeo.post_processing_compute_R_3d_double_grid(check_coarse, check_fine, schurRHS_coarse, schurRHS_fine, ctxt_BCs_Schur, Nx, Ny, Nz, nx_fine_patch, ny_fine_patch, nz_fine_patch, Nib_coarse, Nib_fine, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, delta_layer, Sop_prime, Sop_prime_fine)
# phi_check_coarse = check_coarse_processed[:index_coarse]
# n_p_check_coarse = check_coarse_processed[index_coarse:2*index_coarse]
# n_m_check_coarse = check_coarse_processed[2*index_coarse:3*index_coarse]
# phi_check_fine = check_fine_processed[:index_fine]
# n_p_check_fine = check_fine_processed[index_fine:2*index_fine]
# n_m_check_fine = check_fine_processed[2*index_fine:3*index_fine]

# err_phi_coarse = np.linalg.norm(phi_check_coarse - phi_true_coarse.ravel(order='F')) / np.linalg.norm(phi_true_coarse.ravel(order='F'))
# err_n_p_coarse = np.linalg.norm(n_p_check_coarse - n_p_true_coarse.ravel(order='F')) / np.linalg.norm(n_p_true_coarse.ravel(order='F'))
# err_n_m_coarse = np.linalg.norm(n_m_check_coarse - n_m_true_coarse.ravel(order='F')) / np.linalg.norm(n_m_true_coarse.ravel(order='F'))

# err_phi_fine = np.linalg.norm(phi_check_fine - phi_true_fine.ravel(order='F')) / np.linalg.norm(phi_true_fine.ravel(order='F'))
# err_n_p_fine = np.linalg.norm(n_p_check_fine - n_p_true_fine.ravel(order='F')) / np.linalg.norm(n_p_true_fine.ravel(order='F'))
# err_n_m_fine = np.linalg.norm(n_m_check_fine - n_m_true_fine.ravel(order='F')) / np.linalg.norm(n_m_true_fine.ravel(order='F'))

# print(f'Phi error coarse: {err_phi_coarse}')
# print(f'N_p error coarse: {err_n_p_coarse}')
# print(f'N_m error coarse: {err_n_m_coarse}')
# print(f'Phi error fine: {err_phi_fine}')
# print(f'N_p error fine: {err_n_p_fine}')
# print(f'N_m error fine: {err_n_m_fine}')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X, Y, phi_check_coarse.reshape(Nz, Ny, Nx, order='F'), edgecolor='none')
# ax.set_title("Phi coarse numerical ")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X, Y, phi_true_coarse, edgecolor='none')
# ax.set_title("Phi coarse exact ")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X, Y, n_p_check_coarse.reshape(Nz, Ny, Nx, order='F'), edgecolor='none')
# ax.set_title("N_p coarse numerical ")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X, Y, n_p_true_coarse, edgecolor='none')
# ax.set_title("N_p coarse exact ")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X, Y, n_m_check_coarse.reshape(Nz, Ny, Nx, order='F'), edgecolor='none')
# ax.set_title("N_m coarse numerical ")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X, Y, n_m_true_coarse, edgecolor='none')
# ax.set_title("N_m coarse exact ")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X, Y, phi_check_fine.reshape(Nz, Ny, Nx, order='F'), edgecolor='none')
# ax.set_title("Phi fine numerical ")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X, Y, phi_true_fine, edgecolor='none')
# ax.set_title("Phi fine exact ")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X, Y, n_p_check_fine.reshape(Nz, Ny, Nx, order='F'), edgecolor='none')
# ax.set_title("N_p fine numerical ")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X, Y, n_p_true_fine, edgecolor='none')
# ax.set_title("N_p fine exact ")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X, Y, n_m_check_fine.reshape(Nz, Ny, Nx, order='F'), edgecolor='none')
# ax.set_title("N_m fine numerical ")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X, Y, n_m_true_fine, edgecolor='none')
# ax.set_title("N_m fine exact ")
# ax.set_xlabel("x"); ax.set_ylabel("y")
# plt.show()


# # Check convergence
# RHS_check = b_Op_3d(u_next)
# LHS_check = AxOp_3d(u_next)
# err_curr = np.linalg.norm(LHS_check - RHS_check) / np.linalg.norm(RHS_check)
# print(f'Residual: {err_curr}')

# Build dense Schur matrix
computedRHS_coarse, computedRHS_fine = cpeo.schur_rhs_R_3d_double_grid(b_Op_MMS_coarse(), b_Op_MMS_fine(), Nx, Ny, Nz, nx_fine_patch, ny_fine_patch, nz_fine_patch, Nib_coarse, Nib_fine, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, delta_layer, Jop_prime, Jop_prime_fine)
schurDense_coarse = np.zeros((Nib_coarse * 3, Nib_coarse * 3))
schurDense_fine = np.zeros((Nib_fine * 3, Nib_fine * 3))
for col in range(Nib_coarse * 3): 
    eye_mat = np.zeros(Nib_coarse * 3)
    eye_mat[col] = 1
    p = eye_mat[0:Nib_coarse]
    p_p = eye_mat[Nib_coarse:2*Nib_coarse]
    p_m = eye_mat[2*Nib_coarse:3*Nib_coarse]
    [res_1_coarse, res_2_coarse, res_3_coarse], _ = cpeo.apply_Schur_R_3d_double_grid([p, p_p, p_m], [p, p_p, p_m], ctxt_BCs_Schur, delta_layer, Sop_prime, Sop_prime_fine, Jop_prime, Jop_prime_fine, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, Nx, Ny, Nz, index_coarse)
    schurDense_coarse[:,col] = np.concatenate([res_1_coarse, res_2_coarse, res_3_coarse])

for col in range(Nib_fine * 3): 
    eye_mat = np.zeros(Nib_fine * 3)
    eye_mat[col] = 1
    p = eye_mat[0:Nib_fine]
    p_p = eye_mat[Nib_fine:2*Nib_fine]
    p_m = eye_mat[2*Nib_fine:3*Nib_fine]
    _, [res_1_fine, res_2_fine, res_3_fine] = cpeo.apply_Schur_R_3d_double_grid([p, p_p, p_m], [p, p_p, p_m], ctxt_BCs_Schur, delta_layer, Sop_prime, Sop_prime_fine, Jop_prime, Jop_prime_fine, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, Nx, Ny, Nz, index_coarse)
    schurDense_fine[:,col] = np.concatenate([res_1_fine, res_2_fine, res_3_fine])

# Compute SVD once
U_schur_coarse, Sigma_schur_coarse, Vh_schur_coarse = np.linalg.svd(schurDense_coarse)
U_schur_fine, Sigma_schur_fine, Vh_schur_fine = np.linalg.svd(schurDense_fine)

# Use SVD solve to initialize 
p_next_coarse = cpeo.solve_from_svd(U_schur_coarse, Sigma_schur_coarse, Vh_schur_coarse, computedRHS_coarse)
p_next_fine = cpeo.solve_from_svd(U_schur_fine, Sigma_schur_fine, Vh_schur_fine, computedRHS_fine)
check_coarse_processed, check_fine_processed = cpeo.post_processing_compute_R_3d_double_grid(p_next_coarse, p_next_fine, schurRHS_coarse, schurRHS_fine, ctxt_BCs_Schur, Nx, Ny, Nz, nx_fine_patch, ny_fine_patch, nz_fine_patch, Nib_coarse, Nib_fine, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, delta_layer, Sop_prime, Sop_prime_fine)

# phi_check_coarse = check_coarse_processed[:index_coarse]
# n_p_check_coarse = check_coarse_processed[index_coarse:2*index_coarse]
# n_m_check_coarse = check_coarse_processed[2*index_coarse:3*index_coarse]
# phi_check_fine = check_fine_processed[:index_fine]
# n_p_check_fine = check_fine_processed[index_fine:2*index_fine]
# n_m_check_fine = check_fine_processed[2*index_fine:3*index_fine]

# err_phi_coarse = np.linalg.norm(phi_check_coarse - phi_true_coarse.ravel(order='F')) / np.linalg.norm(phi_true_coarse.ravel(order='F'))
# err_n_p_coarse = np.linalg.norm(n_p_check_coarse - n_p_true_coarse.ravel(order='F')) / np.linalg.norm(n_p_true_coarse.ravel(order='F'))
# err_n_m_coarse = np.linalg.norm(n_m_check_coarse - n_m_true_coarse.ravel(order='F')) / np.linalg.norm(n_m_true_coarse.ravel(order='F'))

# err_phi_fine = np.linalg.norm(phi_check_fine - phi_true_fine.ravel(order='F')) / np.linalg.norm(phi_true_fine.ravel(order='F'))
# err_n_p_fine = np.linalg.norm(n_p_check_fine - n_p_true_fine.ravel(order='F')) / np.linalg.norm(n_p_true_fine.ravel(order='F'))
# err_n_m_fine = np.linalg.norm(n_m_check_fine - n_m_true_fine.ravel(order='F')) / np.linalg.norm(n_m_true_fine.ravel(order='F'))

# print(f'Phi error coarse: {err_phi_coarse}')
# print(f'N_p error coarse: {err_n_p_coarse}')
# print(f'N_m error coarse: {err_n_m_coarse}')
# print(f'Phi error fine: {err_phi_fine}')
# print(f'N_p error fine: {err_n_p_fine}')
# print(f'N_m error fine: {err_n_m_fine}')

G_u_n_coarse, G_u_n_fine = check_coarse_processed, check_fine_processed

u_next_coarse = G_u_n_coarse.copy()
u_next_fine = G_u_n_fine.copy()
G_u_next_coarse = G_u_n_coarse.copy()
G_u_next_fine = G_u_n_fine.copy()
err = []

## Begin profiling here
pr = cProfile.Profile()
pr.enable()

# ##################################
# #######  FULL SYSTEM LOOP  #######
# ##################################
# for its in range(100000):

#######################################
#####  solve R*phi = Rho(u, phi)  #####
#######################################

schurRHS_coarse_computed, schurRHS_fine_computed = b_Op_Schur_3d_double_grid(ctxt_coarse, ctxt_fine)
schurRHS_coarse = b_Op_MMS_coarse() + schurRHS_coarse_computed
schurRHS_fine = b_Op_MMS_fine() + schurRHS_fine_computed
computedRHS_coarse, computedRHS_fine = cpeo.schur_rhs_R_3d_double_grid(schurRHS_coarse, schurRHS_fine, Nx, Ny, Nz, nx_fine_patch, ny_fine_patch, nz_fine_patch, Nib_coarse, Nib_fine, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, delta_layer, Jop_prime, Jop_prime_fine)

# Use SVD to solve
p_next_coarse = cpeo.solve_from_svd(U_schur_coarse, Sigma_schur_coarse, Vh_schur_coarse, computedRHS_coarse)
p_next_fine = cpeo.solve_from_svd(U_schur_fine, Sigma_schur_fine, Vh_schur_fine, computedRHS_fine)
G_u_n_coarse, G_u_n_fine = cpeo.post_processing_compute_R_3d_double_grid(p_next_coarse, p_next_fine, schurRHS_coarse, schurRHS_fine, ctxt_BCs_Schur, Nx, Ny, Nz, nx_fine_patch, ny_fine_patch, nz_fine_patch, Nib_coarse, Nib_fine, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, delta_layer, Sop_prime, Sop_prime_fine)

u_next_coarse = G_u_n_coarse.copy()
u_next_fine = G_u_n_fine.copy()
# u_next = np.concatenate([u_next_coarse, u_next_fine])
# G_u_next = np.concatenate([G_u_n_coarse, G_u_n_fine])
err_coarse = []
err_fine = []

Phi_coarse = u_next_coarse[:index_coarse].reshape((Nz, Ny, Nx), order='F')
Np_coarse = u_next_coarse[index_coarse:2*index_coarse].reshape((Nz, Ny, Nx), order='F')
Nm_coarse = u_next_coarse[2*index_coarse:3*index_coarse].reshape((Nz, Ny, Nx), order='F')

Phi_fine = u_next_fine[:index_fine].reshape((nz_fine_patch, ny_fine_patch, nx_fine_patch), order='F')
Np_fine = u_next_fine[index_fine:2*index_fine].reshape((nz_fine_patch, ny_fine_patch, nx_fine_patch), order='F')
Nm_fine = u_next_fine[2*index_fine:3*index_fine].reshape((nz_fine_patch, ny_fine_patch, nx_fine_patch), order='F')

# Anderson acceleration loop
for inner_its in range(100000):
    schurRHS_coarse_computed, schurRHS_fine_computed = b_Op_Schur_3d_double_grid(u_next_coarse, u_next_fine)
    schurRHS_coarse = b_Op_MMS_coarse() + schurRHS_coarse_computed
    schurRHS_fine = b_Op_MMS_fine() + schurRHS_fine_computed
    schurRHS_coarse = b_Op_MMS_coarse()
    schurRHS_fine = b_Op_MMS_fine()
    computedRHS_coarse, computedRHS_fine = cpeo.schur_rhs_R_3d_double_grid(schurRHS_coarse, schurRHS_fine, Nx, Ny, Nz, nx_fine_patch, ny_fine_patch, nz_fine_patch, Nib_coarse, Nib_fine, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, delta_layer, Jop_prime, Jop_prime_fine)

    # Use SVD to solve
    p_n_coarse = cpeo.solve_from_svd(U_schur_coarse, Sigma_schur_coarse, Vh_schur_coarse, computedRHS_coarse)
    p_n_fine = cpeo.solve_from_svd(U_schur_fine, Sigma_schur_fine, Vh_schur_fine, computedRHS_fine)
    G_u_next_coarse, G_u_next_fine = cpeo.post_processing_compute_R_3d_double_grid(p_n_coarse, p_n_fine, schurRHS_coarse, schurRHS_fine, ctxt_BCs_Schur, Nx, Ny, Nz, nx_fine_patch, ny_fine_patch, nz_fine_patch, Nib_coarse, Nib_fine, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, delta_layer, Sop_prime, Sop_prime_fine)

    m_n = min(m, inner_its + 1)

    # Store differences
    if inner_its < m:
        DU_coarse[:, inner_its] = u_next_coarse - u_n_coarse
        DG_coarse[:, inner_its] = G_u_next_coarse - G_u_n_coarse
        DU_fine[:, inner_its] = u_next_fine - u_n_fine
        DG_fine[:, inner_its] = G_u_next_fine - G_u_n_fine
    else:
        DU_coarse = np.roll(DU_coarse, -1, axis=1)
        DG_coarse = np.roll(DG_coarse, -1, axis=1)
        DU_fine = np.roll(DU_fine, -1, axis=1)
        DG_fine = np.roll(DG_fine, -1, axis=1)
        DU_coarse[:, -1] = u_next_coarse - u_n_coarse
        DG_coarse[:, -1] = G_u_next_coarse - G_u_n_coarse
        DU_fine[:, -1] = u_next_fine - u_n_fine
        DG_fine[:, -1] = G_u_next_fine - G_u_n_fine
    
    f_n_coarse = G_u_next_coarse - u_next_coarse
    f_n_fine = G_u_next_fine - u_next_fine
    DF_coarse = DG_coarse[:, :m_n] - DU_coarse[:, :m_n]
    DF_fine = DG_fine[:, :m_n] - DU_fine[:, :m_n]

    f_n = np.concatenate([f_n_coarse, f_n_fine])
    DF = np.concatenate([DF_coarse, DF_fine], axis=0)
    gamma, _, _, _ = lstsq(DF, f_n)
    
    # # QR decomposition
    # gamma_coarse, residuals, rank, s = lstsq(DF_coarse, f_n_coarse)
    # gamma_fine, residuals, rank, s = lstsq(DF_fine, f_n_fine)
    
    u_n_coarse = u_next_coarse.copy()
    u_n_fine = u_next_fine.copy()
    G_u_n_coarse = G_u_next_coarse.copy()
    G_u_n_fine = G_u_next_fine.copy()
    
    u_next_coarse = (G_u_next_coarse - DG_coarse[:, :m_n] @ gamma) - (1-beta) * (f_n_coarse - DF_coarse @ gamma)
    u_next_fine = (G_u_next_fine - DG_fine[:, :m_n] @ gamma) - (1-beta) * (f_n_fine - DF_fine @ gamma)
    
    # Extract solution components
    Phi_coarse = u_next_coarse[:index_coarse].reshape((Nz, Ny, Nx), order='F')
    Np_coarse = u_next_coarse[index_coarse:2*index_coarse].reshape((Nz, Ny, Nx), order='F')
    Nm_coarse = u_next_coarse[2*index_coarse:3*index_coarse].reshape((Nz, Ny, Nx), order='F')
    p_coarse = u_next_coarse[3*index_coarse:3*index_coarse+Nib_coarse]
    p_p_coarse = u_next_coarse[3*index_coarse+Nib_coarse:3*index_coarse+2*Nib_coarse]
    p_m_coarse = u_next_coarse[3*index_coarse+2*Nib_coarse:]

    Phi_fine = u_next_fine[:index_fine].reshape((nz_fine_patch, ny_fine_patch, nx_fine_patch), order='F')
    Np_fine = u_next_fine[index_fine:2*index_fine].reshape((nz_fine_patch, ny_fine_patch, nx_fine_patch), order='F')
    Nm_fine = u_next_fine[2*index_fine:3*index_fine].reshape((nz_fine_patch, ny_fine_patch, nx_fine_patch), order='F')
    p_fine = u_next_fine[3*index_fine:3*index_fine+Nib_fine]
    p_p_fine = u_next_fine[3*index_fine+Nib_fine:3*index_fine+2*Nib_fine]
    p_m_fine = u_next_fine[3*index_fine+2*Nib_fine:]

    p_next_coarse = u_next_coarse[3*index_coarse:]
    p_next_fine = u_next_fine[3*index_fine:]
    
    # Check convergence
    [res_1_coarse, res_2_coarse, res_3_coarse], [res_1_fine, res_2_fine, res_3_fine] = cpeo.apply_Schur_R_3d_double_grid([p_coarse, p_p_coarse, p_m_coarse], [p_fine, p_p_fine, p_m_fine], ctxt_BCs_Schur, delta_layer, Sop_prime, Sop_prime_fine, Jop_prime, Jop_prime_fine, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, Nx, Ny, Nz, index_coarse)
    
    schur_next_coarse = np.concatenate([res_1_coarse, res_2_coarse, res_3_coarse])
    schur_next_fine = np.concatenate([res_1_fine, res_2_fine, res_3_fine])

    err_curr_coarse = np.linalg.norm(schur_next_coarse - computedRHS_coarse) / np.linalg.norm(computedRHS_coarse)
    err_curr_fine = np.linalg.norm(schur_next_fine - computedRHS_fine) / np.linalg.norm(computedRHS_fine)
    
    err_coarse.append(err_curr_coarse)
    err_fine.append(err_curr_fine)
    
    print(f'Iteration {inner_its}: coarse residual = {err_curr_coarse}')
    print(f'Iteration {inner_its}: fine residual = {err_curr_fine}')
    
    if err_curr_coarse < 5e-3 or err_curr_fine < 5e-3:
        print('Rphi = rho Converged!')
        break
