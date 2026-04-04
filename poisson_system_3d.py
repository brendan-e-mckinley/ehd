
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

## Miscellaneous parameters
tol = 1e-3
beta_BC = 7.94
sigma_bc = 0.78  # 0.68
delta_layer = 0.1  # 5*dx; %6*dx;
cut = 6 * 1.2 * dx # cutoff value

# Anderson acceleration parameters
beta = 0.2
m = 50

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

# Choose number of subdivisions so point spacing matches dx
# Edge length of icosphere ~ rad * 1.0 / nu (approximate)
nu = max(1, int(rad / dx))
vertices, faces = icosphere.icosphere(nu)

# vertices are already on the unit sphere, scale to radius
xib = rad * vertices[:, 0]
yib = rad * vertices[:, 1]
zib = rad * vertices[:, 2]

# Outward unit normals on a sphere are just the unit position vectors
n_x = vertices[:, 0]
n_y = vertices[:, 1]
n_z = vertices[:, 2]

Nib = len(vertices)

fig = plt.figure()
ax = fig.add_subplot(projection='3d') # or fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(xib, yib, zib)

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

## Exact solutions
def Phi_exact(x, y, z):
    return beta_BC * z + 0 * x + 0 * y
def Npm_exact(x, y, z):
    return 0 * x + 1.0

Phi_initial = Phi_exact(X, Y, Z)
Npm_initial = Npm_exact(X, Y, Z)

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
N_pm_guess = Npm_exact(X, Y, Z) #np.zeros_like(X) + 5

## Boundary conditions for Rphi = rho system
Phi_BCs = np.zeros_like(X)
Npm_BCs = np.zeros_like(X)

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
Npm_BCs[0, :, :] = Npm_exact(X[0, :, :], Y[0, :, :], Z[0, :, :])  # z = z_lo
Npm_BCs[-1, :, :] = Npm_exact(X[-1, :, :], Y[-1, :, :], Z[-1, :, :])  # z = z_hi

# x = x_lo and x = x_hi (left and right faces)
Npm_BCs[:, :, 0] = Npm_exact(X[:, :, 0], Y[:, :, 0], Z[:, :, 0])  # x = x_lo
Npm_BCs[:, :, -1] = Npm_exact(X[:, :, -1], Y[:, :, -1], Z[:, :, -1])  # x = x_hi

# y = y_lo and y = y_hi (front and back faces)
Npm_BCs[:, 0, :] = Npm_exact(X[:, 0, :], Y[:, 0, :], Z[:, 0, :])  # y = y_lo
Npm_BCs[:, -1, :] = Npm_exact(X[:, -1, :], Y[:, -1, :], Z[:, -1, :])  # y = y_hi

# Boundary conditions context for Schur solve of Rphi = rho 
ctxt_BCs_Schur = np.concatenate([
    Phi_BCs.ravel(order='F'),
    Npm_BCs.ravel(order='F'),
    Npm_BCs.ravel(order='F'),
    np.zeros(len(xib)) - (sigma_bc),
    np.zeros(len(xib)),
    np.zeros(len(xib))
])

# Boundary conditions context for full Rphi = rho system
ctxt_BCs = np.concatenate([
    Phi_BCs.ravel(order='F'),
    Npm_BCs.ravel(order='F'),
    Npm_BCs.ravel(order='F'),
    np.zeros(len(xib)) - (sigma_bc/delta_layer),
    np.zeros(len(xib)),
    np.zeros(len(xib))
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

ctxt = np.concatenate([
    Phi_guess.ravel(order='F'),
    N_pm_guess.ravel(order='F'),
    N_pm_guess.ravel(order='F'),
    np.zeros(Nib),
    np.zeros(Nib),
    np.zeros(Nib)
])

# Extract solution components
index = Nx * Ny * Nz
Phi = ctxt[:index].reshape((Nz, Ny, Nx), order='F')
Np = ctxt[index:2*index].reshape((Nz, Ny, Nx), order='F')
Nm = ctxt[2*index:3*index].reshape((Nz, Ny, Nx), order='F')

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
def delta_a(r, a):
    return (1/(2*np.pi*a**2)) * np.exp(-0.5*(r/a)**2)

@jit(nopython=True)
def delta(r):
    return delta_a(r, 1.2*dx)

@jit(nopython=True)
def delta_r(r):
    return (1/(1.2*dx))**2 * r * delta_a(r, 1.2*dx)

def Sop_prime(q):
    return cpeo.spreadQ_prime_3d(X, Y, Z, xib, yib, zib, n_x, n_y, n_z, q, delta_r, cut, dx, dy, dz)

def Jop(P):
    return cpeo.interpPhi_3d(X, Y, Z, xib, yib, zib, P, delta, cut, dx, dy, dz)

def Jop_prime(P):
    return cpeo.interpPhi_prime_3d(X, Y, Z, xib, yib, zib, n_x, n_y, n_z, P, delta_r, cut, dx, dy, dz)

def G_d_G_3d(Phi, N_pm):
    return cpeo.Grad_dot_Grad_3d(Phi, N_pm, dx, dy, dz, Nx, Ny, Nz)

# def G_d_G(Phi, N_pm):
#     return cpeo.Grad_dot_Grad(Phi, N_pm, dx, dy, Nx, Ny, Phi_BC, N_pm_BC)

def b_Op_3d(ctxt):
    print(f"b_Op_3d input norm: {np.linalg.norm(ctxt)}")
    return cpeo.Build_RHS_3d(ctxt, ctxt_BCs, Lap, G_d_G_3d, delta_layer, Nx, Ny, Nz, Nib, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, Jop, Jop_prime)

def b_Op_Schur_3d(ctxt):
    return cpeo.Build_RHS_Schur_System_3d(ctxt, ctxt_BCs_Schur, G_d_G_3d, delta_layer, Nx, Ny, Nz, Nib, Jop, Jop_prime, Sop_prime, dx, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)

def AxOp_3d(ctxt):
    return cpeo.Constrained_Lap_3d(ctxt, ctxt_BCs, delta_layer, Nx, Ny, Nz, Nib, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, Sop_prime, Jop_prime)

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
n_test = np.sin(X) * np.sin(Y) * np.sin(Z)
f = -2 * np.sin(X) * np.sin(Y) * np.sin(Z)
bcs = 20 + np.zeros_like(X)
bcs[1:-1,1:-1,1:-1] = np.zeros_like(Xint)
zero_bcs = np.zeros_like(X)

#\nabla^2 phi + n = f
#\implies phi + \nabla^{-2} n &= \nabla^{-2} f

# quick check for manufactured solution
lap_phi = -3.0 * np.sin(X) * np.sin(Y) * np.sin(Z)
phi_computed = amr_solve.solve_poisson(f, bcs, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, 0.3) - amr_solve.solve_poisson(n_test, zero_bcs, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, 0.3)

residual_lin = np.linalg.norm(phi_computed - phi_test) / np.linalg.norm(phi_test)
print(f'Linearity test residual: {residual_lin}')

## DO/UNDO LAP 
lap_phi_num = amr_solve.apply_poisson(phi_test, bcs, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
phi_from_lap = amr_solve.solve_poisson(lap_phi_num, bcs, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, 0.3) - amr_solve.solve_poisson(n_test, zero_bcs, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, 0.3)

residual_phi = np.linalg.norm(phi_from_lap - phi_test) / np.linalg.norm(phi_from_lap)
print(f'Do/undo residual: {residual_phi}')

## G DOT G
phi_test = 5 + np.sin(X) * np.sin(Y) * np.sin(Z)
n_p_test = 5 + np.sin(X) * np.sin(Y) * np.sin(Z)

# Analytical solution
gdg = (np.cos(X)**2 * np.sin(Y)**2 * np.sin(Z)**2 +
       np.sin(X)**2 * np.cos(Y)**2 * np.sin(Z)**2 +
       np.sin(X)**2 * np.sin(Y)**2 * np.cos(Z)**2).ravel(order='F')

# Numerical solution
gdg_num = G_d_G_3d(phi_test.ravel(order='F'), n_p_test.ravel(order='F'))

# Residual
residual = np.linalg.norm(gdg_num - gdg) / np.linalg.norm(gdg)
print(f'Residual: {residual}')

## Test basic system 
RHS = b_Op_3d(ctxt)
shape = 3 * Nx * Ny * Nz + 3 * Nib
AxOp = AxLinOp_3d(shape)

shape = 3 * Nib

# compare initial residual
LHS = AxOp(ctxt)
err_curr = np.linalg.norm(LHS - RHS) / np.linalg.norm(RHS)
print(f'Initial residual: {err_curr}')

check_solve, _ = gmres(AxOp, RHS, rtol=tol, restart=500, callback=lambda rk: print(f"GMRES residual: {np.linalg.norm(rk)}"))

Phi_initial = check_solve[:index]
Np_initial = check_solve[index:2*index]
Nm_initial = check_solve[2*index:3*index]

# Create a structured grid
grid = pv.StructuredGrid(X, Y, Z)

# Add the scalar fields
grid['Phi_exact'] = Phi_initial
grid['Np_exact'] = Np_initial
grid['Nm_exact'] = Nm_initial

# Plot Phi_exact
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars='Phi_exact', show_scalar_bar=True)
plotter.show(title='Phi_exact')

# Plot Npm_exact
plotter = pv.Plotter()
# Get the scalar values
scalars = grid['Np_exact']

# Create opacity array: transparent if |value - 1| < 0.001, else opaque
opacity = np.where(np.abs(scalars - 1) < 0.001, 0.0, 1.0)

plotter = pv.Plotter()
plotter.add_mesh(grid, scalars='Np_exact', opacity=opacity, show_scalar_bar=True)
plotter.show(title='Np_exact')

# Plot Npm_exact
plotter = pv.Plotter()
# Get the scalar values
scalars = grid['Nm_exact']

# Create opacity array: transparent if |value - 1| < 0.001, else opaque
opacity = np.where(np.abs(scalars - 1) < 0.001, 0.0, 1.0)

plotter = pv.Plotter()
plotter.add_mesh(grid, scalars='Nm_exact', opacity=opacity, show_scalar_bar=True)
plotter.show(title='Nm_exact')

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

##########################
#####  SOLVER SETUP  #####
##########################

## Initialize variables for Rphi = rho solve
schurRHS = b_Op_Schur_3d(ctxt)
DU = np.full((len(schurRHS), m), np.nan)
DG = np.full((len(schurRHS), m), np.nan)

u_n = ctxt.copy()

# Build dense Schur matrix
# schurOp = cpeo.SchurLinearOperator_R_3d(shape, Nib, delta_layer, Sop_prime, Jop_prime, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
schurRHS = b_Op_Schur_3d(ctxt)
computedRHS = cpeo.schur_rhs_R_3d(schurRHS, ctxt_BCs_Schur, Nx, Ny, Nz, Nib, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, delta_layer, Jop_prime)
schurDense = np.zeros((Nib * 3, Nib * 3))
for col in range(Nib * 3): 
    eye_mat = np.zeros(Nib * 3)
    eye_mat[col] = 1
    p = eye_mat[0:Nib]
    p_p = eye_mat[Nib:2*Nib]
    p_m = eye_mat[2*Nib:3*Nib]
    res_1, res_2, res_3 = cpeo.apply_Schur_R_3d([p, p_p, p_m], ctxt_BCs_Schur, delta_layer, Sop_prime, Jop_prime, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, Nx, Ny, Nz, index)
    schurDense[:,col] = np.concatenate([res_1, res_2, res_3])

# Compute SVD once
U_schur, Sigma_schur, Vh_schur = np.linalg.svd(schurDense)

# Use SVD solve to initialize 
p_next = cpeo.solve_from_svd(U_schur, Sigma_schur, Vh_schur, computedRHS)
G_u_n = cpeo.post_processing_compute_R_3d(p_next, schurRHS, Nx, Ny, Nz, Nib, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, delta_layer, Sop_prime)

# compare solved residual
LHS = AxOp(G_u_n)
RHS = b_Op_3d(G_u_n)
err_curr = np.linalg.norm(LHS - RHS) / np.linalg.norm(RHS)
print(f'Solved residual: {err_curr}')

u_next = G_u_n.copy()
G_u_next = G_u_n.copy()
err = []

# # compute body forces 
# Phi_full = np.zeros((Ny+2, Nx+2))
# Np_full = np.ones((Ny+2, Nx+2))
# Nm_full = np.ones((Ny+2, Nx+2))

# Phi_full[1:-1, 1:-1] = Phi
# Phi_full[1:-1, 0] = Phi[:, -1]
# Phi_full[1:-1, -1] = Phi[:, -1]
# Phi_full[0, :] = -25 # TODO: HACKY AND WRONG, FIX THIS
# Phi_full[-1, :] = 25 # TODO: HACKY AND WRONG, FIX THIS
# Np_full[1:-1, 1:-1] = Np
# Nm_full[1:-1, 1:-1] = Nm

# Grad_x_Phi_Flat = (G_x_full @ Phi_full.ravel(order='F'))
# Grad_y_Phi_Flat = (G_y_full @ Phi_full.ravel(order='F'))

# Np_relevant_y = Np_full[1:-1,:]
# Nm_relevant_y = Nm_full[1:-1,:]
# Np_relevant_x = Np_full[:,1:-1]
# Nm_relevant_x = Nm_full[:,1:-1]

# bodyForces_x = -(Np_relevant_x.ravel(order='F') - Nm_relevant_x.ravel(order='F')) / (2 * delta_layer**2) * Grad_x_Phi_Flat
# bodyForces_y = -(Np_relevant_y.ravel(order='F') - Nm_relevant_y.ravel(order='F')) / (2 * delta_layer**2) * Grad_y_Phi_Flat

# body_x = bodyForces_x.reshape(Ny + 2, Nx, order='F')
# body_y = bodyForces_y.reshape(Ny, Nx + 2, order='F')
# body_interpolated_x = 0.5 * (body_x[:-1, :] + body_x[1:, :])
# body_interpolated_y = 0.5 * (body_y[:, :-1] + body_y[:, 1:])

# ## Initialize variables for Nu = F solve
# f = body_interpolated_x.ravel(order='F')
# g = body_interpolated_y.ravel(order='F')

# f_bc = f.ravel(order='F') + f_bc_mat.ravel(order='F')
# g_bc = g.ravel(order='F') + g_bc_mat.ravel(order='F')

# # Build dense Schur matrix for hydrodynamic system
# schurDense_N = np.zeros((Nib * 2, Nib * 2))
# for col in range(Nib * 2): 
#     eye_mat_N = np.zeros(Nib * 2)
#     eye_mat_N[col] = 1
#     lam_X = eye_mat_N[0:Nib]
#     lam_Y = eye_mat_N[Nib:]
#     schurDense_N[:,col] = stokes.apply_Schur_new(lam_X, lam_Y, stokes_LU, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, cut)

# U_schur_fluid, Sigma_schur_fluid, Vh_schur_fluid = np.linalg.svd(schurDense_N)

# U, V, P, lam_X, lam_Y = stokes.solve_factorized(-L/2, L/2, stokes_LU, U_schur_fluid, Sigma_schur_fluid, Vh_schur_fluid, f_bc, g_bc, h_bc, z_x, z_y, rad, Nx, tol, cut)
# #U, V, P, lam_X, lam_Y = stokes.solve(-L/2, L/2, f_bc, g_bc, h_bc, z_x, z_y, rad, Nx, tol, cut)

# Uplot = U.reshape((Ny + 1, Nx), order='F')
# Vplot = V.reshape((Ny, Nx + 1), order='F')
# Pplot = P.reshape((Ny + 1, Nx + 1), order='F')

# U_interpolated = np.zeros([Ny, Nx])
# V_interpolated = np.zeros([Ny, Nx])
# for col_index in range(Uplot.shape[1]):
#     col = Uplot[:, col_index]
#     for row_index in range(len(col) - 1):
#         midpoint = (col[row_index] + col[row_index+1]) / 2
#         U_interpolated[row_index, col_index] = midpoint


# for row_index in range(Vplot.shape[0]):
#     row = Vplot[row_index, :]
#     for col_index in range(len(row) - 1):
#         midpoint = (row[col_index] + row[col_index+1]) / 2
#         V_interpolated[row_index, col_index] = midpoint

# UFull = np.zeros((Ny + 2, Nx + 2))
# UFull[1:Ny + 1, 1:Nx + 1] = U_interpolated

# VFull = np.zeros((Ny + 2, Nx + 2))
# VFull[1:Ny + 1, 1:Nx + 1] = V_interpolated

# # Get initial fluid velocity values
# U_fluid = (UFull[1:Ny+1,1:Nx+1]).ravel(order='F')
# V_fluid = (VFull[1:Ny+1,1:Nx+1]).ravel(order='F')

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

schurRHS = b_Op_Schur_3d(ctxt)
computedRHS = cpeo.schur_rhs_R_3d(schurRHS, ctxt_BCs_Schur, Nx, Ny, Nz, Nib, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, delta_layer, Jop_prime)

# Use SVD to solve
p_next = cpeo.solve_from_svd(U_schur, Sigma_schur, Vh_schur, computedRHS)
G_u_n = cpeo.post_processing_compute_R_3d(p_next, schurRHS, Nx, Ny, Nz, Nib, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, delta_layer, Sop_prime)

u_next = G_u_n.copy()
G_u_next = G_u_n.copy()
err = []

Phi = u_next[:index].reshape((Nz, Ny, Nx), order='F')
Np = u_next[index:2*index].reshape((Nz, Ny, Nx), order='F')
Nm = u_next[2*index:3*index].reshape((Nz, Ny, Nx), order='F')

# Create a structured grid
grid = pv.StructuredGrid(X, Y, Z)

# Add the scalar fields
grid['Phi_solved'] = Phi.ravel(order='F')
grid['Np_solved'] = Np.ravel(order='F')
grid['Nm_solved'] = Nm.ravel(order='F')

# Plot Phi_solved
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars='Phi_solved', show_scalar_bar=True)
plotter.show(title='Phi_solved')

# Plot Np_solved
plotter = pv.Plotter()
mesh = plotter.add_mesh(grid, scalars='Np_solved', show_scalar_bar=True)
# opacity = [0.0 if val == 1 else 1.0 for val in np.unique(Np)]
# mesh.set_scalar_bar_range([Np.min(), Np.max()])
# mesh.map_scalars('Np_solved', clim=[Np.min(), Np.max()])
# mesh.set_opacity(opacity)
plotter.show(title='Np_solved')

# Plot Nm_solved
plotter = pv.Plotter()
mesh = plotter.add_mesh(grid, scalars='Nm_solved', show_scalar_bar=True)
# opacity = [0.0 if val == 1 else 1.0 for val in np.unique(Nm)]
# mesh.set_scalar_bar_range([Nm.min(), Nm.max()])
# mesh.map_scalars('Nm_solved', clim=[Nm.min(), Nm.max()])
# mesh.set_opacity(opacity)
plotter.show(title='Nm_solved')

# Anderson acceleration loop
for inner_its in range(100000):
    schurRHS = b_Op_Schur_3d(ctxt)
    computedRHS = cpeo.schur_rhs_R_3d(schurRHS, ctxt_BCs_Schur, Nx, Ny, Nz, Nib, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, delta_layer, Jop_prime)

    # Use SVD to solve
    p_n = cpeo.solve_from_svd(U_schur, Sigma_schur, Vh_schur, computedRHS)
    G_u_next = cpeo.post_processing_compute_R_3d(p_next, schurRHS, Nx, Ny, Nz, Nib, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, delta_layer, Sop_prime)

    m_n = min(m, inner_its + 1)
    
    # Store differences
    if inner_its < m:
        DU[:, inner_its] = u_next - u_n
        DG[:, inner_its] = G_u_next - G_u_n
    else:
        DU = np.roll(DU, -1, axis=1)
        DG = np.roll(DG, -1, axis=1)
        DU[:, -1] = u_next - u_n
        DG[:, -1] = G_u_next - G_u_n
    
    f_n = G_u_next - u_next
    DF = DG[:, :m_n] - DU[:, :m_n]
    
    # QR decomposition
    gamma, residuals, rank, s = lstsq(DF, f_n)
    
    u_n = u_next.copy()
    G_u_n = G_u_next.copy()
    
    u_next = (G_u_next - DG[:, :m_n] @ gamma) - (1-beta) * (f_n - DF @ gamma)
    
    # Extract solution components
    Phi = u_next[:index].reshape((Nz, Ny, Nx), order='F')
    Np = u_next[index:2*index].reshape((Nz, Ny, Nx), order='F')
    Nm = u_next[2*index:3*index].reshape((Nz, Ny, Nx), order='F')
    p = u_next[3*index:3*index+Nib]
    p_p = u_next[3*index+Nib:3*index+2*Nib]
    p_m = u_next[3*index+2*Nib:]

    p_next = u_next[3*index:]
    
    # Check convergence
    res1, res2, res3 = cpeo.apply_Schur_R_3d([p, p_p, p_m], ctxt_BCs_Schur, delta_layer, Sop_prime, Jop_prime, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, Nx, Ny, Nz, index)
    schur_next = np.concatenate([res1, res2, res3])
    err_curr = np.linalg.norm(schur_next - computedRHS) / np.linalg.norm(computedRHS)
    
    err.append(err_curr)
    
    print(f'Iteration {inner_its}: residual = {err_curr}')
    
    if err_curr < 5e-3:
        print('Rphi = rho Converged!')
        break

# Create a structured grid
grid = pv.StructuredGrid(X, Y, Z)

# Add the scalar fields
grid['Phi_solved'] = Phi.ravel(order='F')
grid['Np_solved'] = Np.ravel(order='F')
grid['Nm_solved'] = Nm.ravel(order='F')

# Plot Phi_solved
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars='Phi_solved', show_scalar_bar=True)
plotter.show(title='Phi_solved')

# Plot Np_solved
plotter = pv.Plotter()
mesh = plotter.add_mesh(grid, scalars='Np_solved', show_scalar_bar=True)
# opacity = [0.0 if val == 1 else 1.0 for val in np.unique(Np)]
# mesh.set_scalar_bar_range([Np.min(), Np.max()])
# mesh.map_scalars('Np_solved', clim=[Np.min(), Np.max()])
# mesh.set_opacity(opacity)
plotter.show(title='Np_solved')

# Plot Nm_solved
plotter = pv.Plotter()
mesh = plotter.add_mesh(grid, scalars='Nm_solved', show_scalar_bar=True)
# opacity = [0.0 if val == 1 else 1.0 for val in np.unique(Nm)]
# mesh.set_scalar_bar_range([Nm.min(), Nm.max()])
# mesh.map_scalars('Nm_solved', clim=[Nm.min(), Nm.max()])
# mesh.set_opacity(opacity)
plotter.show(title='Nm_solved')
    
    # ctxt = u_next.copy()
    
    # # compute body forces 
    # Phi_full = np.zeros((Ny+2, Nx+2))
    # Np_full = np.ones((Ny+2, Nx+2))
    # Nm_full = np.ones((Ny+2, Nx+2))

    # Phi_full[1:-1, 1:-1] = Phi
    # Phi_full[1:-1, 0] = Phi[:, -1]
    # Phi_full[1:-1, -1] = Phi[:, -1]
    # Phi_full[0, 1:-1] = -25
    # Phi_full[-1, 1:-1] = 25
    # Np_full[1:-1, 1:-1] = Np
    # Nm_full[1:-1, 1:-1] = Nm

    # Grad_x_Phi_Flat = (G_x_full @ Phi_full.ravel(order='F'))
    # Grad_y_Phi_Flat = (G_y_full @ Phi_full.ravel(order='F'))

    # Np_relevant_y = Np_full[1:-1,:]
    # Nm_relevant_y = Nm_full[1:-1,:]
    # Np_relevant_x = Np_full[:,1:-1]
    # Nm_relevant_x = Nm_full[:,1:-1]

    # bodyForces_x = -(Np_relevant_x.ravel(order='F') - Nm_relevant_x.ravel(order='F')) / (2 * delta_layer**2) * Grad_x_Phi_Flat
    # bodyForces_y = -(Np_relevant_y.ravel(order='F') - Nm_relevant_y.ravel(order='F')) / (2 * delta_layer**2) * Grad_y_Phi_Flat

    # body_x = bodyForces_x.reshape(Ny + 2, Nx, order='F')
    # body_y = bodyForces_y.reshape(Ny, Nx + 2, order='F')
    # body_interpolated_x = 0.5 * (body_x[:-1, :] + body_x[1:, :])
    # body_interpolated_y = 0.5 * (body_y[:, :-1] + body_y[:, 1:])

    # ################################
    # #####  solve N*u = F(phi)  #####
    # ################################

    # # RHS
    # f = body_interpolated_x.ravel(order='F')
    # g = body_interpolated_y.ravel(order='F')

    # f_bc = f.ravel(order='F') + f_bc_mat.ravel(order='F')
    # g_bc = g.ravel(order='F') + g_bc_mat.ravel(order='F')

    # U, V, P, lam_X, lam_Y = stokes.solve_factorized(-L/2, L/2, stokes_LU, U_schur_fluid, Sigma_schur_fluid, Vh_schur_fluid, f_bc, g_bc, h_bc, z_x, z_y, rad, Nx, tol, cut)
    # #U, V, P, lam_X, lam_Y = stokes.solve(-L/2, L/2, f_bc, g_bc, h_bc, z_x, z_y, rad, Nx, tol, cut)

    # Uplot = U.reshape((Ny + 1, Nx), order='F')
    # Vplot = V.reshape((Ny, Nx + 1), order='F')
    # Pplot = P.reshape((Ny + 1, Nx + 1), order='F')

    # U_interpolated = np.zeros([Ny, Nx])
    # V_interpolated = np.zeros([Ny, Nx])
    # for col_index in range(Uplot.shape[1]):
    #     col = Uplot[:, col_index]
    #     for row_index in range(len(col) - 1):
    #         midpoint = (col[row_index] + col[row_index+1]) / 2
    #         U_interpolated[row_index, col_index] = midpoint

    # for row_index in range(Vplot.shape[0]):
    #     row = Vplot[row_index, :]
    #     for col_index in range(len(row) - 1):
    #         midpoint = (row[col_index] + row[col_index+1]) / 2
    #         V_interpolated[row_index, col_index] = midpoint

    # UFull = np.zeros((Ny + 2, Nx + 2))
    # UFull[1:Ny + 1, 1:Nx + 1] = U_interpolated

    # VFull = np.zeros((Ny + 2, Nx + 2))
    # VFull[1:Ny + 1, 1:Nx + 1] = V_interpolated

    # U_fluid = (UFull[1:Ny+1,1:Nx+1]).ravel(order='F')
    # V_fluid = (VFull[1:Ny+1,1:Nx+1]).ravel(order='F')

    # residual_check_RHS = b_Op(u_next, U_fluid, V_fluid)
    # residual_check_AxOp = AxOp(u_next)
    # residual_check = np.linalg.norm(residual_check_AxOp - residual_check_RHS) / np.linalg.norm(residual_check_RHS)
    # print(f'New residual Rphi = {residual_check}')

    # if residual_check < 1e-4:
    #     print('Full system converged!')
    #     break

# check Nu residual 
Nu_RHS = np.concatenate([f_bc, g_bc, h_bc, z_x, z_y])
u_tilde = np.concatenate([U, V, P, lam_X, lam_Y])
Nu = stokes.apply_A(u_tilde, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x_staggered, G_y_staggered, D_x_staggered, D_y_staggered, cut)
residual_check_N = np.linalg.norm(Nu - Nu_RHS) / np.linalg.norm(Nu_RHS)
print(f'New residual Nu = {residual_check_N}')

# check system residual
full_system_operator_applied = np.concatenate([Nu, residual_check_AxOp])
full_system_RHS = np.concatenate([Nu_RHS, residual_check_RHS])
residual_check_full = np.linalg.norm(full_system_operator_applied - full_system_RHS) / np.linalg.norm(full_system_RHS)
print(f'Full residual = {residual_check_full}')

# end profiling
pr.disable()
pr.dump_stats("profile_fast.prof")

###################################
#######  SAVE/PLOT RESULTS  #######
###################################

# # Save results
# savemat('Full_System_Results_Block_Lap.mat', {
#     'ctxt_Rphi': ctxt,
#     'u_next': u_next,
#     'Xint': Xint,
#     'Yint': Yint,
#     'xib': xib,
#     'yib': yib,
#     'Phi': Phi,
#     'Np': Np,
#     'Nm': Nm,
#     'err': np.array(err),
#     'U_fluid': UFull,
#     'V_fluid': VFull,
#     'P_fluid': Pplot,
#     'lam_X': lam_X,
#     'lam_Y': lam_Y
# })

# Define circular mask (radius 0.25 centered at origin)
radius = 0.25
mask = (X**2 + Y**2) <= radius**2

# Apply mask to UFull and VFull
UFull[mask] = np.nan
VFull[mask] = np.nan

plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, UFull, VFull, color='red', density=5, linewidth=1, arrowsize=1.5)
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Flow Field Streamline Plot')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(True)

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, UFull, cmap=cmap, edgecolor='none')
ax.set_title("U")
ax.set_xlabel("x"); ax.set_ylabel("y")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, VFull, cmap=cmap, edgecolor='none')
ax.set_title("V")
ax.set_xlabel("x"); ax.set_ylabel("y")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, np.sqrt(UFull**2 + VFull**2), cmap=cmap, edgecolor='none')
ax.set_title("|velocity|")
ax.set_xlabel("x"); ax.set_ylabel("y")

plt.show()
