
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
Nx, Ny, Nz = 128, 128, 128  # 256; % number of grid points along one direction
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
delta_layer = 5*dx  # 0.1 # 5*dx; %6*dx;
cut_coarse = 6 * 1.2 * dx # cutoff value

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
    #return np.sin(x) * np.sin(y) * np.sin(z)
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
    np.zeros(Nib_coarse),
    np.zeros(Nib_coarse),
    np.zeros(Nib_coarse)
])

# Boundary conditions context for full Rphi = rho system
ctxt_BCs = np.concatenate([
    Phi_BCs.ravel(order='F'),
    Np_BCs.ravel(order='F'),
    Nm_BCs.ravel(order='F'),
    np.zeros(Nib_coarse),
    np.zeros(Nib_coarse),
    np.zeros(Nib_coarse)
])

####################
## EXACT SOLUTION ##
####################
r_coarse = np.sqrt(X**2 + Y**2 + Z**2)
rib = np.sqrt(xib_coarse**2 + yib_coarse**2 + zib_coarse**2)
phi_true_coarse = np.sin(r_coarse)
#phi_true_coarse = np.sin(X) * np.sin(Y) * np.sin(Z)
n_p_true_coarse = np.exp(-np.sin(r_coarse))
n_m_true_coarse = np.exp(np.sin(r_coarse))

ctxt_true_coarse = np.concatenate([
    phi_true_coarse.ravel(order='F'),
    n_p_true_coarse.ravel(order='F'),
    n_m_true_coarse.ravel(order='F'),
    np.full_like(xib_coarse, np.nan),
    np.full_like(xib_coarse, np.nan),
    np.full_like(xib_coarse, np.nan)
])

ctxt_coarse = np.concatenate([
    Phi_guess.ravel(order='F'),
    N_p_guess.ravel(order='F'),
    N_m_guess.ravel(order='F'),
    np.zeros(Nib_coarse),
    np.zeros(Nib_coarse),
    np.zeros(Nib_coarse)
])

guess = np.zeros(Nib_coarse * 3)

# Extract solution components
index_coarse = Nx * Ny * Nz
Phi_coarse = ctxt_coarse[:index_coarse].reshape((Nz, Ny, Nx), order='F')
Np_coarse = ctxt_coarse[index_coarse:2*index_coarse].reshape((Nz, Ny, Nx), order='F')
Nm_coarse = ctxt_coarse[2*index_coarse:3*index_coarse].reshape((Nz, Ny, Nx), order='F')

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
    return -(1/(1.2*dx)**2) * r * delta_a_3d(r, 1.2*dx)

def Sop_prime(q):
    return cpeo.spreadQ_prime_3d(X, Y, Z, xib_coarse, yib_coarse, zib_coarse, n_x_coarse, n_y_coarse, n_z_coarse, q, delta_r_coarse, cut_coarse, dx, dy, dz)

def Jop(P):
    return cpeo.interpPhi_3d(X, Y, Z, xib_coarse, yib_coarse, zib_coarse, P, delta_coarse, cut_coarse, dx, dy, dz)

def Jop_prime(P):
    return cpeo.interpPhi_prime_3d(X, Y, Z, xib_coarse, yib_coarse, zib_coarse, n_x_coarse, n_y_coarse, n_z_coarse, P, delta_r_coarse, cut_coarse, dx, dy, dz)

def G_d_G_3d(Phi, N_pm):
    return cpeo.Grad_dot_Grad_3d(Phi, N_pm, dx, dy, dz, Nx, Ny, Nz)

def b_Op_3d(ctxt):
    print(f"b_Op_3d input norm: {np.linalg.norm(ctxt)}")
    return cpeo.Build_RHS_3d(ctxt, ctxt_BCs, Lap, G_d_G_3d, delta_layer, Nx, Ny, Nz, Nib_coarse, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, Jop, Jop_prime)

def b_Op_Schur_3d(ctxt):
    return cpeo.Build_RHS_Schur_System_3d(ctxt, ctxt_BCs_Schur, G_d_G_3d, delta_layer, Nx, Ny, Nz, Nib_coarse, Jop, Jop_prime, Sop_prime, dx, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)

def b_Op_MMS_coarse():
    return cpeo.Build_RHS_Schur_System_Manufactured_Solution_3d(ctxt_true_coarse, X, Y, Z, xib_coarse, yib_coarse, zib_coarse, Nx, Ny, Nz, Nib_coarse, delta_layer)

def AxOp_3d(ctxt):
    return cpeo.Constrained_Lap_3d(ctxt, ctxt_BCs, delta_layer, Nx, Ny, Nz, Nib_coarse, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, Sop_prime, Jop_prime)

def AxLinOp_3d(shape):
    def mv(vec):
        return AxOp_3d(vec)
        
    return LinearOperator((shape, shape), matvec=mv)

# lap_exact = -3 * np.sin(X) *np.sin(Y) * np.sin(Z)

# bcs = np.zeros_like(X)
# bcs[1:-1,1:-1,1:-1] = np.zeros_like(Xint)
# computed_lap = amr_solve.apply_poisson_single_grid(phi_true_coarse, bcs, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
# lap_err = np.linalg.norm(computed_lap - lap_exact) / np.linalg.norm(lap_exact)

# print(f"Lap error: {lap_err}")

# compare RHSs
true_RHS = b_Op_MMS_coarse()
computed_RHS = b_Op_Schur_3d(ctxt_true_coarse)

RHS_Np = (computed_RHS[index_coarse:2*index_coarse]).reshape(Nz, Ny, Nx, order='F')
RHS_Np_true = (true_RHS[index_coarse:2*index_coarse]).reshape(Nz, Ny, Nx, order='F')
RHS_Np_rel = RHS_Np[1:-1,1:-1,1:-1]
RHS_Np_true_rel = RHS_Np_true[1:-1,1:-1,1:-1]

RHS_Nm = (computed_RHS[2*index_coarse:3*index_coarse]).reshape(Nz, Ny, Nx, order='F')
RHS_Nm_true = (true_RHS[2*index_coarse:3*index_coarse]).reshape(Nz, Ny, Nx, order='F')
RHS_Nm_rel = RHS_Nm[1:-1,1:-1,1:-1]
RHS_Nm_true_rel = RHS_Nm_true[1:-1,1:-1,1:-1]

err_RHS_Np = np.linalg.norm(RHS_Np_rel.ravel(order='F') - RHS_Np_true_rel.ravel(order='F')) / np.linalg.norm(RHS_Np_true_rel.ravel(order='F'))
err_RHS_Nm = np.linalg.norm(RHS_Nm_rel.ravel(order='F') - RHS_Nm_true_rel.ravel(order='F')) / np.linalg.norm(RHS_Nm_true_rel.ravel(order='F'))
err_RHS_p = np.linalg.norm(computed_RHS[3*index_coarse:3*index_coarse+Nib_coarse] - true_RHS[3*index_coarse:3*index_coarse+Nib_coarse]) / np.linalg.norm(true_RHS[3*index_coarse:3*index_coarse+Nib_coarse])
err_RHS_pp = np.linalg.norm(computed_RHS[3*index_coarse+Nib_coarse:3*index_coarse+2*Nib_coarse] - true_RHS[3*index_coarse+Nib_coarse:3*index_coarse+2*Nib_coarse]) / np.linalg.norm(true_RHS[3*index_coarse+Nib_coarse:3*index_coarse+2*Nib_coarse])
err_RHS_pm = np.linalg.norm(computed_RHS[3*index_coarse+2*Nib_coarse:] - true_RHS[3*index_coarse+2*Nib_coarse:]) / np.linalg.norm(true_RHS[3*index_coarse+2*Nib_coarse:])

print(f'RHS error Np: {err_RHS_Np}')
print(f'RHS error Nm: {err_RHS_Nm}')
print(f'RHS error p: {err_RHS_p}')
print(f'RHS error pp: {err_RHS_pp}')
print(f'RHS error pm: {err_RHS_pm}')

##########################
#####  SOLVER SETUP  #####
##########################

## Initialize variables for Rphi = rho solve
schurRHS_coarse = b_Op_MMS_coarse()

DU_coarse = np.full((len(schurRHS_coarse), m), np.nan)
DG_coarse = np.full((len(schurRHS_coarse), m), np.nan)

u_n_coarse = ctxt_coarse.copy()

# Build dense Schur matrix
computedRHS_coarse = cpeo.schur_rhs_R_3d(b_Op_MMS_coarse(), ctxt_BCs_Schur, Nx, Ny, Nz, Nib_coarse, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, delta_layer, Jop_prime)
schurDense_coarse = np.zeros((Nib_coarse * 3, Nib_coarse * 3))
for col in range(Nib_coarse * 3): 
    eye_mat = np.zeros(Nib_coarse * 3)
    eye_mat[col] = 1
    p = eye_mat[0:Nib_coarse]
    p_p = eye_mat[Nib_coarse:2*Nib_coarse]
    p_m = eye_mat[2*Nib_coarse:3*Nib_coarse]
    [res_1_coarse, res_2_coarse, res_3_coarse] = cpeo.apply_Schur_R_3d([p, p_p, p_m], ctxt_BCs_Schur, delta_layer, Sop_prime, Jop_prime, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, Nx, Ny, Nz, index_coarse)
    schurDense_coarse[:,col] = np.concatenate([res_1_coarse, res_2_coarse, res_3_coarse])

# Compute SVD once
U_schur_coarse, Sigma_schur_coarse, Vh_schur_coarse = np.linalg.svd(schurDense_coarse)

# Use SVD solve to initialize 
p_next_coarse = cpeo.solve_from_svd(U_schur_coarse, Sigma_schur_coarse, Vh_schur_coarse, computedRHS_coarse)
check_coarse_processed = cpeo.post_processing_compute_R_3d(p_next_coarse, schurRHS_coarse, ctxt_BCs_Schur, Nx, Ny, Nz, Nib_coarse, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, delta_layer, Sop_prime)

phi_check_coarse = check_coarse_processed[:index_coarse]
n_p_check_coarse = check_coarse_processed[index_coarse:2*index_coarse]
n_m_check_coarse = check_coarse_processed[2*index_coarse:3*index_coarse]

err_phi_coarse = np.linalg.norm(phi_check_coarse - phi_true_coarse.ravel(order='F')) / np.linalg.norm(phi_true_coarse.ravel(order='F'))
err_n_p_coarse = np.linalg.norm(n_p_check_coarse - n_p_true_coarse.ravel(order='F')) / np.linalg.norm(n_p_true_coarse.ravel(order='F'))
err_n_m_coarse = np.linalg.norm(n_m_check_coarse - n_m_true_coarse.ravel(order='F')) / np.linalg.norm(n_m_true_coarse.ravel(order='F'))

print(f'Phi error coarse: {err_phi_coarse}')
print(f'N_p error coarse: {err_n_p_coarse}')
print(f'N_m error coarse: {err_n_m_coarse}')
