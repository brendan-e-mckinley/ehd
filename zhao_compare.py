
import numpy as np
import pyvista as pv
import cmasher as cmr
import cProfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from numba import jit
from sksparse.cholmod import cholesky
from scipy.sparse import spdiags, eye, kron, diags, csr_matrix, bmat, lil_matrix, coo_matrix
from scipy.sparse.linalg import splu, eigs, spsolve, gmres, LinearOperator
from scipy.linalg import qr, lstsq
from scipy.io import loadmat, savemat
from scipy.interpolate import Akima1DInterpolator, interpn
import CPEO_utils_zhao_compare as cpeo
import stokes_solver_utils_open_BCs as stokes
from matplotlib.colors import ListedColormap, Normalize

###########################
######  PARAMETERS  #######
###########################

## Grid parameters
Nx = 512 # 256; % number of grid points along one direction
L = 4.0 * np.pi 
x = np.linspace(-L/2, L/2, Nx+2) 
dx = x[1] - x[0]
y = x.copy()
dy = y[1] - y[0]

## Miscellaneous parameters
tol = 1e-4
beta_BC = 10 / L
sigma_bc = 0.1  # 0.68
delta_layer = 40 * dx #0.1#; %6*dx;
cut = 6 * 1.2 * dx # cutoff value

# Anderson acceleration parameters
beta = 0.2
m = 50

# time parameters
N_t = 1
dt = 0.01

##########################
######  GRID SETUP  ######
##########################

## Nodal grid (interior points)
xint = x[1:-1]
yint = y[1:-1]
Ny = len(yint)

X, Y = np.meshgrid(x, y)
Xint, Yint = np.meshgrid(xint, yint)

## Staggered grid (horizontal faces for V, vertical faces for U, cell centers for P)
N_U = (Nx + 2) * (Ny + 1)
N_V = (Nx + 1) * (Ny + 2)
N_P = (Nx + 1) * (Ny + 1)
x_mid = x + dx / 2
y_mid = y + dy / 2
x_offset = x_mid[:-1]
y_offset = y_mid[:-1]

UGridX, UGridY = np.meshgrid(x, y_offset) # UGridX, UGridY
VGridX, VGridY = np.meshgrid(x_offset, y)
PGridX, PGridY = np.meshgrid(x_offset, y_offset)

# ## Immersed boundary
# rad = 1
# particle_x_offset = 0
# particle_y_offset = 2 * np.pi - 10 * delta_layer - rad
# dth = dx / rad
# theta = np.arange(0, 2*np.pi, dth)
# Nib = len(theta)
# xib = rad * np.cos(theta) 
# yib = -(particle_y_offset) + rad * np.sin(theta) #-(np.pi + 2) + 
# n_x = np.cos(theta)
# n_y = np.sin(theta)

## Immersed boundary
rad = 1
particle_x_offset = 0
particle_y_offset = 4 * np.pi - 8 * delta_layer - rad

dth_approx = dx / rad
Nib = int(round(2 * np.pi / dth_approx))
if Nib % 2 == 1:
    Nib += 1          # force even, so theta and pi-theta always land on the lattice

dth = 2 * np.pi / Nib  # exact spacing, no leftover partial arc

theta = np.arange(Nib) * dth
xib = rad * np.cos(theta) + particle_x_offset
yib = rad * np.sin(theta) - particle_y_offset
n_x = np.cos(theta)
n_y = np.sin(theta)

# reflect xib about particle_x_offset and see if the point set matches itself
xib_reflected = 2*particle_x_offset - xib
# for each reflected point, is there a matching (xib, yib) pair?
diffs = np.min(np.abs(xib[:,None] - xib_reflected[None,:]) + np.abs(yib[:,None] - yib[None,:]), axis=1)
print(np.max(diffs))

#########################
######  OPERATORS  ######
#########################

## Fluid Laplacian operators
# Staggered laplacians (dirichlet boundary conditions in y, dirichlet boundary conditions in x)
Lap_U, Lap_V = stokes.build_staggered_Laps_do_nothing(Nx, Ny, dx, dy)

## ELECTROSTATIC LAPLACIANS W/ NEUMANN CONDITIONS

# Phi laplacian: Neumann conditions
e_y = (1/dy**2) * np.ones(Ny)
D2_d_y_phi = spdiags([e_y, -2*e_y, e_y], [-1, 0, 1], Ny, Ny)

dok_mat_y_phi = D2_d_y_phi.todok()
dok_mat_y_phi[-1, -1] = -(1/dy**2)
D2_d_y_phi = dok_mat_y_phi.tocsr()

e_x = (1/dx**2) * np.ones(Nx) 
D2_d_x_phi = spdiags([e_x, -2*e_x, e_x], [-1, 0, 1], Nx, Nx)

dok_mat_x_phi = D2_d_x_phi.todok()
dok_mat_x_phi[0, 0] = -(1/dx**2)
dok_mat_x_phi[-1, -1] = -(1/dx**2)
D2_d_x_phi = dok_mat_x_phi.tocsr()

I_nx = eye(Nx)
I_ny = eye(Ny)
Lap_phi = -(kron(I_nx, D2_d_y_phi) + kron(D2_d_x_phi, I_ny))
dLap_phi = cholesky(Lap_phi)

rhs_phi = np.zeros([Nx, Ny])
rhs_phi[-1, :] = beta_BC / dx
print(rhs_phi.ravel(order='F'))

phi_solved = -spsolve(Lap_phi, -rhs_phi.ravel(order='F'))
phi = phi_solved.reshape(Ny,Nx, order='F')

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xint, Yint, phi, cmap=cmap, edgecolor='none')
ax.set_title("Phi")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.show()

computed_rhs = Lap_phi @ phi.ravel(order='F')

residual = computed_rhs - rhs_phi.ravel(order='F')
print(residual)

# N_pm laplacian: Neumann conditions
D2_d_y_npm = spdiags([e_y, -2*e_y, e_y], [-1, 0, 1], Ny, Ny)

dok_mat_y_npm = D2_d_y_npm.todok()
# # neumann = 0:
# dok_mat_y_npm[-1, -1] = -(1/dy**2)
D2_d_y_npm = dok_mat_y_npm.tocsr()

D2_d_x_npm = spdiags([e_x, -2*e_x, e_x], [-1, 0, 1], Nx, Nx)

dok_mat_x_npm = D2_d_x_npm.todok()
dok_mat_x_npm[0, 0] = -(1/dy**2)
dok_mat_x_npm[-1, -1] = -(1/dy**2)
D2_d_x_npm = dok_mat_x_npm.tocsr()

Lap_npm = -(kron(I_nx, D2_d_y_npm) + kron(D2_d_x_npm, I_ny))
dLap_npm = cholesky(Lap_npm)

rhs_test_npm = np.zeros([Nx, Ny])
rhs_test_npm[-1, :] = 1 / (dy**2)
print(rhs_test_npm.ravel(order='F'))

npm_solved = -spsolve(Lap_npm, -rhs_test_npm.ravel(order='F'))
npm = npm_solved.reshape(Ny,Nx, order='F')

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xint, Yint, npm, cmap=cmap, edgecolor='none')
ax.set_title("Npm")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.show()

computed_rhs = Lap_npm @ npm.ravel(order='F')

residual = computed_rhs - rhs_test_npm.ravel(order='F')
print(residual)

D_x_1d = diags(
    [-0.5/dx * np.ones(Nx+2), 0.5/dx * np.ones(Nx+2)],
    offsets=[-1, 1],
    shape=(Nx+2, Nx+2)
).tocsr()#[1:-1, :]

D_y_1d = diags(
    [-0.5/dy * np.ones(Ny+2), 0.5/dy * np.ones(Ny+2)],
    offsets=[-1, 1],
    shape=(Ny+2, Ny+2)
).tocsr()#[1:-1, :]
# D_y_1d[0, 0] = -D_y_1d[0, 0] * 2 # Dirichlet = 0

G_x_full = kron(D_x_1d, eye(Ny+2, format='csr'), format='csr')
G_y_full = kron(eye(Nx+2, format='csr'), D_y_1d, format='csr')

G_x_U = kron(D_x_1d[1:-1, :], eye(Ny + 1, format='csr'), format='csr')
G_y_V = kron(eye(Nx + 1, format='csr'), D_y_1d[1:-1, :], format='csr')

print(G_x_U.toarray())
print(G_y_V.toarray())

# Staggered gradient and divergence operators
G_x_staggered, G_y_staggered, D_x_staggered, D_y_staggered = stokes.build_staggered_Grads_Divs_do_nothing(Nx, Ny, dx, dy)

## Prefactor big Stokes operator
Z_UV = csr_matrix((N_U, N_V))
Z_VU = csr_matrix((N_V, N_U))
Z_PP = csr_matrix((N_P, N_P))

# Saddle point system
big_L = bmat([
    [Lap_U, Z_VU,  -G_x_staggered],
    [Z_UV,  Lap_V, -G_y_staggered],
    [D_x_staggered,   D_y_staggered,   Z_PP] 
], format='csr')

stokes_LU = splu(big_L)

#############################################
#######  BOUNDARY/INITIAL CONDITIONS  #######
#############################################

## Exact solutions
def Phi_exact(x, y):
    return beta_BC * (y + L/2) + 0 * x
def Npm_exact(x, y):
    return 0 * x + 1.0
def Np_electrode(phi_1, np_1):
    return (np_1) / (1 + phi_1) 
def Nm_electrode(phi_1, nm_1):
    return (nm_1) / (1 - phi_1)

## Initial conditions for Rphi = rho system
ld = loadmat('zhao_compare.mat')
METHOD = 'cubic'  # equivalent to 'makima' in MATLAB

Ny_ld = 256
Nx_ld = 256
Nib_ld = int(len(ld['xib']))
sz = Ny_ld * Nx_ld

V_rigid_ld = ld['V_rigid']
ctxt_ld = ld['ctxt_Rphi'].ravel(order='F')
Phi_ld = ctxt_ld[:sz].reshape(Ny_ld, Nx_ld, order='F')
N_p_ld = ctxt_ld[sz:2*sz].reshape(Ny_ld, Nx_ld, order='F')
N_m_ld = ctxt_ld[2*sz:3*sz].reshape(Ny_ld, Nx_ld, order='F')
Q_ld = ctxt_ld[3*sz:3*sz+Nib_ld]
Q_p_ld = ctxt_ld[3*sz+Nib_ld:3*sz+2*Nib_ld]
Q_m_ld = ctxt_ld[3*sz+2*Nib_ld:3*sz+3*Nib_ld]

# symmetry
print('Phi asym:', np.max(np.abs(Phi_ld - np.fliplr(Phi_ld))) / np.max(np.abs(Phi_ld)))
print('Np asym:', np.max(np.abs(N_p_ld - np.fliplr(N_p_ld))) / np.max(np.abs(N_p_ld)))
print('Nm asym:', np.max(np.abs(N_m_ld - np.fliplr(N_m_ld))) / np.max(np.abs(N_m_ld)))

Xint_ld = ld['Xint']
Yint_ld = ld['Yint']

# Extract the coordinate vectors from the loaded grid
x_ld = Xint_ld[0, :]  # First row gives x-coordinates
y_ld = Yint_ld[:, 0]  # First column gives y-coordinates

# Interpolate initial guesses
Phi_init = interpn((x_ld, y_ld), Phi_ld, (Xint.T, Yint.T), method='linear', bounds_error=False, fill_value=None)
N_p_init = interpn((x_ld, y_ld), N_p_ld, (Xint.T, Yint.T), method='nearest', bounds_error=False, fill_value=None)
N_m_init = interpn((x_ld, y_ld), N_m_ld, (Xint.T, Yint.T), method='nearest', bounds_error=False, fill_value=None)

# symmetrize initial seed
Phi_init = 0.5*(Phi_init + np.fliplr(Phi_init))
N_p_init = 0.5*(N_p_init + np.fliplr(N_p_init))
N_m_init = 0.5*(N_m_init + np.fliplr(N_m_init))

Q_init = np.zeros(Nib)
Q_p_init = np.zeros(Nib)
Q_m_init = np.zeros(Nib)

# initial guess for electrode
electrode_p = Np_electrode(Phi_init[0,:], N_p_init[0,:])
electrode_m = Nm_electrode(Phi_init[0,:], N_m_init[0,:])

## Boundary conditions for Rphi = rho system
Phi_BCs = np.zeros_like(Xint)
Np_BCs = np.zeros_like(Xint)
Nm_BCs = np.zeros_like(Xint)

Phi_BCs[-1,:] += (1/dy/dy) * beta_BC * dy
Np_BCs[0,:] += (1/dy/dy) * electrode_p#Phi_init[0,:], N_p_init[0,:]) #np.zeros_like(xint), np.ones_like(xint))#Phi_init[0,:], N_p_init[0,:]) #np.zeros_like(xint), np.ones_like(xint)
Np_BCs[-1,:] += (1/dy/dy) * np.ones_like(xint)
Nm_BCs[0,:] += (1/dy/dy) * electrode_m#Phi_init[0,:], N_m_init[0,:])#Phi_init[0,:], N_m_init[0,:]) #np.zeros_like(xint), np.ones_like(xint)
Nm_BCs[-1,:] += (1/dy/dy) * np.ones_like(xint)

# Boundary conditions context for Schur solve of Rphi = rho 
ctxt_BCs_Schur = np.concatenate([
    Phi_BCs.ravel(order='F'),
    Np_BCs.ravel(order='F'),
    Nm_BCs.ravel(order='F'),
    rad * np.sin(theta), #np.zeros(len(xib)) - (sigma_bc),
    np.zeros(len(xib)),
    np.zeros(len(xib))
])

# Boundary conditions context for full Rphi = rho system
ctxt_BCs = np.concatenate([
    Phi_BCs.ravel(order='F'),
    Np_BCs.ravel(order='F'),
    Nm_BCs.ravel(order='F'),
    (rad * np.sin(theta)) / delta_layer, #np.zeros(len(xib)) - (sigma_bc/delta_layer),
    np.zeros(len(xib)),
    np.zeros(len(xib))
])

ctxt = np.concatenate([
    Phi_init.ravel(order='F'),
    N_p_init.ravel(order='F'),
    N_m_init.ravel(order='F'),
    Q_init,
    Q_p_init,
    Q_m_init
])

Phi = ctxt[:Ny*Nx].reshape((Ny, Nx), order='F')
Np = ctxt[Ny*Nx:2*Nx*Ny].reshape((Ny, Nx), order='F')
Nm = ctxt[2*Ny*Nx:3*Nx*Ny].reshape((Ny, Nx), order='F')

# ## Start ion concentrations at equilibrium
# Np_prev = np.ones(Ny*Nx)
# Nm_prev = np.ones(Ny*Nx)

## Boundary conditions for Nu = F system
f_bc_mat = np.zeros((Ny + 1, Nx + 2))
g_bc_mat = np.zeros((Ny + 2, Nx + 1))
h_bc = np.zeros((Ny + 1, Nx + 1)).ravel(order='F')
z_bc = np.zeros(2 * Nib)
V_bc = np.zeros(3)

## Initial conditions for Nu = F system
U_fluid = np.zeros((Ny + 1) * (Ny + 2))
V_fluid = np.zeros((Ny + 2) * (Ny + 1))

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
    return cpeo.spreadQ_prime(Xint, Yint, xib, yib, n_x, n_y, q, delta_r, cut)

def Jop(P):
    return cpeo.interpPhi(Xint, Yint, xib, yib, P, delta, cut)

def Jop_prime(P):
    return cpeo.interpPhi_prime(Xint, Yint, xib, yib, n_x, n_y, P, delta_r, cut)

def G_d_G_p(Phi, N_p, electrode):
    return cpeo.Grad_dot_Grad_neumann_far_field(Phi, N_p, dx, dy, Nx, Ny, beta_BC, electrode)

def G_d_G_m(Phi, N_m, electrode):
    return cpeo.Grad_dot_Grad_neumann_far_field(Phi, N_m, dx, dy, Nx, Ny, beta_BC, electrode)

def b_Op_Schur(ctxt, bcs, U_fluid, V_fluid, electrode_p, electrode_m): 
    return cpeo.Build_RHS_Schur_System_neumann(ctxt, electrode_p, electrode_m, bcs, U_fluid, V_fluid, Lap_phi, G_d_G_p, G_d_G_m, delta_layer, Nx, Ny, Nib, Jop, Jop_prime, dx, beta_BC)

def b_Op(ctxt, bcs, U_fluid, V_fluid, electrode_p, electrode_m):
    return cpeo.Build_RHS_rho_neumann(ctxt, electrode_p, electrode_m, bcs, U_fluid, V_fluid, dLap_phi, dLap_npm, Lap_phi, G_d_G_p, G_d_G_m, delta_layer, Nx, Ny, Nib, Jop, Jop_prime, dx, beta_BC)

def AxOp(ctxt):
    return cpeo.Constrained_Lap_neumann(ctxt, dLap_phi, dLap_npm, delta_layer, Nx, Ny, Nib, Sop_prime, Jop_prime)

delta_x, delta_y = stokes.make_composite_deltas(dx, n=3)

##########################
#####  SOLVER SETUP  #####
##########################

## Initialize variables for Rphi = rho solve
schurRHS = b_Op_Schur(ctxt, ctxt_BCs_Schur, U_fluid, V_fluid, electrode_p, electrode_m)
DU = np.full((len(schurRHS), m), np.nan)
DG = np.full((len(schurRHS), m), np.nan)

u_n = ctxt.copy()
p_guess = u_n[3*Nx*Ny:]

# Build dense Schur matrix
schurRHS = b_Op_Schur(ctxt, ctxt_BCs_Schur, U_fluid, V_fluid, electrode_p, electrode_m)
computedRHS = cpeo.schur_rhs_R_neumann(dLap_phi, dLap_npm, schurRHS, Nx, Ny, Nib, delta_layer, Jop_prime)
schurDense = np.zeros((Nib * 3, Nib * 3))
for col in range(Nib * 3): 
    eye_mat = np.zeros(Nib * 3)
    eye_mat[col] = 1
    p = eye_mat[0:Nib]
    p_p = eye_mat[Nib:2*Nib]
    p_m = eye_mat[2*Nib:3*Nib]
    res_1, res_2, res_3 = cpeo.apply_Schur_R_neumann(dLap_phi, dLap_npm, [p, p_p, p_m], delta_layer, Nx, Ny, Sop_prime, Jop_prime)
    schurDense[:,col] = np.concatenate([res_1, res_2, res_3])

# Compute SVD once
U_schur, Sigma_schur, Vh_schur = np.linalg.svd(schurDense)

# Use SVD solve to initialize 
p_next = cpeo.solve_from_svd(U_schur, Sigma_schur, Vh_schur, computedRHS)
G_u_n = cpeo.post_processing_compute_R_neumann(dLap_phi, dLap_npm, p_next, schurRHS, Nx, Ny, Nib, delta_layer, Sop_prime)
u_next = G_u_n.copy()
G_u_next = G_u_n.copy()
err = []

# compute body forces 
Phi_full = np.zeros((Ny+2, Nx+2))
Np_full = np.ones((Ny+2, Nx+2))
Nm_full = np.ones((Ny+2, Nx+2))

Phi_full[1:-1, 1:-1] = Phi
Phi_full[1:-1, 0] = Phi[:, 0]
Phi_full[1:-1, -1] = Phi[:, -1]
Phi_full[-1, 1:-1] = dx * beta_BC + Phi[-1, :]

# new, maybe wrong
Phi_full[-1, 0] = dx * beta_BC + Phi[-1, 0]
Phi_full[-1, -1] = dx * beta_BC + Phi[-1, -1]

Np_full[1:-1, 1:-1] = Np
Nm_full[1:-1, 1:-1] = Nm

# new, maybe wrong 
Np_full[1:-1, 0] = Np[:, 0]
Np_full[1:-1, -1] = Np[:, -1]
Nm_full[1:-1, 0] = Nm[:, 0]
Nm_full[1:-1, -1] = Nm[:, -1]

electrode_p = Np_electrode(Phi[0,:], Np[0,:])
electrode_m = Nm_electrode(Phi[0,:], Nm[0,:])

Np_full[0, 1:-1] = electrode_p
Nm_full[0, 1:-1] = electrode_m

# interpolate onto U grid (average in y) and find grad_x phi
phi_interpolated_U = 0.5 * (Phi_full[:-1, :] + Phi_full[1:, :])   # shape (Ny+1, Nx+2)
Grad_x_Phi_U_int = (G_x_U @ phi_interpolated_U.ravel(order='F')).reshape(Ny + 1, Nx, order='F')
Grad_x_Phi_U = np.zeros_like(phi_interpolated_U)
Grad_x_Phi_U[:, 1:-1] = Grad_x_Phi_U_int

# interpolate onto V grid (average in x) and find grad_y phi
phi_interpolated_V = 0.5 * (Phi_full[:, :-1] + Phi_full[:, 1:])   # shape (Ny+2, Nx+1)
Grad_y_Phi_V_int = (G_y_V @ phi_interpolated_V.ravel(order='F')).reshape(Ny, Nx + 1, order='F')
Grad_y_Phi_V = np.zeros_like(phi_interpolated_V)
Grad_y_Phi_V[1:-1, :] = Grad_y_Phi_V_int
Grad_y_Phi_V[0, :] = phi_interpolated_V[1, :] / dx
Grad_y_Phi_V[-1, :] = beta_BC

Np_interpolated_U = 0.5 * (Np_full[:-1, :] + Np_full[1:, :])
Nm_interpolated_U = 0.5 * (Nm_full[:-1, :] + Nm_full[1:, :])
Np_interpolated_V = 0.5 * (Np_full[:, :-1] + Np_full[:, 1:])
Nm_interpolated_V = 0.5 * (Nm_full[:, :-1] + Nm_full[:, 1:])

bodyForces_x = -(Np_interpolated_U.ravel(order='F') - Nm_interpolated_U.ravel(order='F')) / (2 * delta_layer**2) * Grad_x_Phi_U.ravel(order='F')
bodyForces_y = -(Np_interpolated_V.ravel(order='F') - Nm_interpolated_V.ravel(order='F')) / (2 * delta_layer**2) * Grad_y_Phi_V.ravel(order='F')

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(UGridX, UGridY, bodyForces_x.reshape(Ny + 1, Nx + 2, order='F'), cmap=cmap, edgecolor='none')
ax.set_title("x body forces")
ax.set_xlabel("x"); ax.set_ylabel("y")

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(VGridX, VGridY, bodyForces_y.reshape(Ny + 2, Nx + 1, order='F'), cmap=cmap, edgecolor='none')
ax.set_title("y body forces")
ax.set_xlabel("x"); ax.set_ylabel("y")

plt.show()

## Initialize variables for Nu = F solve
f = bodyForces_x.ravel(order='F')
g = bodyForces_y.ravel(order='F')

f_bc = f.ravel(order='F') + f_bc_mat.ravel(order='F')
g_bc = g.ravel(order='F') + g_bc_mat.ravel(order='F')

# Build dense Schur matrix for hydrodynamic system
schurDense_N = np.zeros((Nib * 2 + 3, Nib * 2 + 3))
for col in range(Nib * 2 + 3): 
    eye_mat_N = np.zeros(Nib * 2 + 3)
    eye_mat_N[col] = 1
    lam = eye_mat_N[0:2*Nib]
    V_rigid = eye_mat_N[2*Nib:]
    schurDense_N[:,col] = stokes.LHS_op_big_K(lam, V_rigid, stokes_LU, UGridX, UGridY, VGridX, VGridY, xib, yib, particle_x_offset, particle_y_offset, delta_x, delta_y, N_U, N_V, N_P, Nib, cut)

U_schur_fluid, Sigma_schur_fluid, Vh_schur_fluid = np.linalg.svd(schurDense_N)

U, V, P, lam_X, lam_Y, V_rigid = stokes.solve_factorized_K(UGridX, UGridY, VGridX, VGridY, stokes_LU, U_schur_fluid, Sigma_schur_fluid, Vh_schur_fluid, f_bc, g_bc, h_bc, z_bc, V_bc, xib, yib, Nib, Nx, Ny, dx, tol, cut)

U_fluid = U.reshape((Ny + 1, Nx + 2), order='F')
V_fluid = V.reshape((Ny + 2, Nx + 1), order='F')
V_fluid[0, :] = 0

# Pplot = P.reshape((Ny + 1, Nx + 1), order='F')

# ## Begin profiling here
# pr = cProfile.Profile()
# pr.enable()

##################################
#######  FULL SYSTEM LOOP  #######
##################################

for t_step in range(N_t):
    # ##################################
    # #######  single time step  #######
    # ##################################
    for its in range(100000):
        #######################################
        #####  solve R*phi = Rho(u, phi)  #####
        #######################################

        schurRHS = b_Op_Schur(ctxt, ctxt_BCs_Schur, U_fluid, V_fluid, electrode_p, electrode_m)
        computedRHS = cpeo.schur_rhs_R_neumann(dLap_phi, dLap_npm, schurRHS, Nx, Ny, Nib, delta_layer, Jop_prime)

        # Use SVD to solve
        p_next = cpeo.solve_from_svd(U_schur, Sigma_schur, Vh_schur, computedRHS)
        G_u_n = cpeo.post_processing_compute_R_neumann(dLap_phi, dLap_npm, p_next, schurRHS, Nx, Ny, Nib, delta_layer, Sop_prime)

        u_next = G_u_n.copy()
        G_u_next = G_u_n.copy()
        err = []

        # Anderson acceleration loop
        for inner_its in range(100000):
            schurRHS = b_Op_Schur(u_next, ctxt_BCs_Schur, U_fluid, V_fluid, electrode_p, electrode_m)
            computedRHS = cpeo.schur_rhs_R_neumann(dLap_phi, dLap_npm, schurRHS, Nx, Ny, Nib, delta_layer, Jop_prime)
            
            # Use SVD to solve
            p_n = cpeo.solve_from_svd(U_schur, Sigma_schur, Vh_schur, computedRHS)
            G_u_next = cpeo.post_processing_compute_R_neumann(dLap_phi, dLap_npm, p_n, schurRHS, Nx, Ny, Nib, delta_layer, Sop_prime)

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
            Phi = u_next[:Ny*Nx].reshape((Ny, Nx), order='F')
            Np = u_next[Ny*Nx:2*Nx*Ny].reshape((Ny, Nx), order='F')
            Nm = u_next[2*Ny*Nx:3*Nx*Ny].reshape((Ny, Nx), order='F')
            p = u_next[3*Nx*Ny:3*Nx*Ny+Nib]
            p_p = u_next[3*Nx*Ny+Nib:3*Nx*Ny+2*Nib]
            p_m = u_next[3*Nx*Ny+2*Nib:]

            p_next = u_next[3*Nx*Ny:]
            
            # Check convergence
            res1, res2, res3 = cpeo.apply_Schur_R_neumann(dLap_phi, dLap_npm, [p, p_p, p_m], delta_layer, Nx, Ny, Sop_prime, Jop_prime)
            schur_next = np.concatenate([res1, res2, res3])
            err_curr = np.linalg.norm(schur_next - computedRHS) / np.linalg.norm(computedRHS)
            
            err.append(err_curr)
            
            print(f'Iteration {inner_its}: residual = {err_curr}')
            
            if err_curr < 1e-4:
                print('Rphi = rho Converged!')
                break

            ## Boundary conditions for Rphi = rho system
            Phi_BCs = np.zeros_like(Xint)
            Np_BCs = np.zeros_like(Xint)
            Nm_BCs = np.zeros_like(Xint)

            # update electrode
            electrode_p = Np_electrode(Phi[0,:], Np[0,:])
            electrode_m = Nm_electrode(Phi[0,:], Nm[0,:])

            Phi_BCs[-1,:] += (1/dy/dy) * beta_BC * dy
            Np_BCs[0,:] += (1/dy/dy) * electrode_p
            Np_BCs[-1,:] += (1/dy/dy) * np.ones_like(xint)
            Nm_BCs[0,:] += (1/dy/dy) * electrode_m
            Nm_BCs[-1,:] += (1/dy/dy) * np.ones_like(xint)

            # Boundary conditions context for Schur solve of Rphi = rho 
            ctxt_BCs_Schur = np.concatenate([
                Phi_BCs.ravel(order='F'),
                Np_BCs.ravel(order='F'),
                Nm_BCs.ravel(order='F'),
                np.zeros(len(xib)) - (sigma_bc),
                np.zeros(len(xib)),
                np.zeros(len(xib))
            ])

            # Boundary conditions context for full Rphi = rho system
            ctxt_BCs = np.concatenate([
                Phi_BCs.ravel(order='F'),
                Np_BCs.ravel(order='F'),
                Nm_BCs.ravel(order='F'),
                np.zeros(len(xib)) - (sigma_bc/delta_layer),
                np.zeros(len(xib)),
                np.zeros(len(xib))
            ])

        ctxt = u_next.copy()

        # rebuild dense Schur matrix with updated BCs
        schurRHS = b_Op_Schur(ctxt, ctxt_BCs_Schur, U_fluid, V_fluid, electrode_p, electrode_m)
        computedRHS = cpeo.schur_rhs_R_neumann(dLap_phi, dLap_npm, schurRHS, Nx, Ny, Nib, delta_layer, Jop_prime)
        
        # compute body forces 
        Phi_full = np.zeros((Ny+2, Nx+2))
        Np_full = np.ones((Ny+2, Nx+2))
        Nm_full = np.ones((Ny+2, Nx+2))

        Phi_full[1:-1, 1:-1] = Phi
        Phi_full[1:-1, 0] = Phi[:, 0]
        Phi_full[1:-1, -1] = Phi[:, -1]
        Phi_full[-1, 1:-1] = dx * beta_BC + Phi[-1, :]

        # new, maybe wrong
        Phi_full[-1, 0] = dx * beta_BC + Phi[-1, 0]
        Phi_full[-1, -1] = dx * beta_BC + Phi[-1, -1]

        Np_full[1:-1, 1:-1] = Np
        Nm_full[1:-1, 1:-1] = Nm

        # new, maybe wrong 
        Np_full[1:-1, 0] = Np[:, 0]
        Np_full[1:-1, -1] = Np[:, -1]
        Nm_full[1:-1, 0] = Nm[:, 0]
        Nm_full[1:-1, -1] = Nm[:, -1]

        # update electrode
        electrode_p = Np_electrode(Phi[0,:], Np[0,:])
        electrode_m = Nm_electrode(Phi[0,:], Nm[0,:])

        Np_full[0, 1:-1] = electrode_p
        Nm_full[0, 1:-1] = electrode_m

        Np_full[0, 0] = Np_full[0, 1]
        Np_full[0, -1] = Np_full[0, -2]
        Nm_full[0, 0] = Nm_full[0, 1]
        Nm_full[0, -1] = Nm_full[0, -2]

        # interpolate onto U grid (average in y) and find grad_x phi
        phi_interpolated_U = 0.5 * (Phi_full[:-1, :] + Phi_full[1:, :])   # shape (Ny+1, Nx+2)
        Grad_x_Phi_U_int = (G_x_U @ phi_interpolated_U.ravel(order='F')).reshape(Ny + 1, Nx, order='F')
        Grad_x_Phi_U = np.zeros_like(phi_interpolated_U)
        Grad_x_Phi_U[:, 1:-1] = Grad_x_Phi_U_int

        # interpolate onto V grid (average in x) and find grad_y phi
        phi_interpolated_V = 0.5 * (Phi_full[:, :-1] + Phi_full[:, 1:])   # shape (Ny+2, Nx+1)
        Grad_y_Phi_V_int = (G_y_V @ phi_interpolated_V.ravel(order='F')).reshape(Ny, Nx + 1, order='F')
        Grad_y_Phi_V = np.zeros_like(phi_interpolated_V)
        Grad_y_Phi_V[1:-1, :] = Grad_y_Phi_V_int
        Grad_y_Phi_V[0, :] = phi_interpolated_V[1, :] / dx
        Grad_y_Phi_V[-1, :] = beta_BC

        Np_interpolated_U = 0.5 * (Np_full[:, :-1] + Np_full[:, 1:])
        Nm_interpolated_U = 0.5 * (Nm_full[:, :-1] + Nm_full[:, 1:])
        Np_interpolated_V = 0.5 * (Np_full[:-1, :] + Np_full[1:, :])
        Nm_interpolated_V = 0.5 * (Nm_full[:-1, :] + Nm_full[1:, :])

        bodyForces_x = -(Np_interpolated_U.ravel(order='F') - Nm_interpolated_U.ravel(order='F')) / (2 * delta_layer**2) * Grad_x_Phi_U.ravel(order='F')
        bodyForces_y = -(Np_interpolated_V.ravel(order='F') - Nm_interpolated_V.ravel(order='F')) / (2 * delta_layer**2) * Grad_y_Phi_V.ravel(order='F')

        ################################
        #####  solve N*u = F(phi)  #####
        ################################

        # RHS
        f = bodyForces_x.ravel(order='F')
        g = bodyForces_y.ravel(order='F')

        f_bc = f.ravel(order='F') + f_bc_mat.ravel(order='F')
        g_bc = g.ravel(order='F') + g_bc_mat.ravel(order='F')

        U, V, P, lam_X, lam_Y, V_rigid = stokes.solve_factorized_K(UGridX, UGridY, VGridX, VGridY, stokes_LU, U_schur_fluid, Sigma_schur_fluid, Vh_schur_fluid, f_bc, g_bc, h_bc, z_bc, V_bc, xib, yib, Nib, Nx, Ny, dx, tol, cut)
        #U, V, P, lam_X, lam_Y = stokes.solve(-L/2, L/2, f_bc, g_bc, h_bc, z_x, z_y, rad, Nx, tol, cut)

        Nu_RHS = np.concatenate([f_bc, g_bc, h_bc, z_bc, V_bc])
        lam = stokes.interleave(lam_X, lam_Y)
        u_tilde = np.concatenate([U, V, P, lam, V_rigid])
        K = stokes.build_K(xib, yib, [particle_x_offset, particle_y_offset], Nib)
        Nu = stokes.apply_A_K(u_tilde, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x_staggered, G_y_staggered, D_x_staggered, D_y_staggered, K, cut)
        residual_check_N = np.linalg.norm(Nu - Nu_RHS) / np.linalg.norm(Nu_RHS)
        print(f'Current residual Nu = {residual_check_N}')

        U_fluid = U.reshape((Ny + 1, Nx + 2), order='F')
        V_fluid = V.reshape((Ny + 2, Nx + 1), order='F')
        V_fluid[0, :] = 0

        residual_check_RHS = b_Op(u_next, ctxt_BCs, U_fluid, V_fluid, electrode_p, electrode_m)
        residual_check_AxOp = AxOp(u_next)
        residual_check = np.linalg.norm(residual_check_AxOp - residual_check_RHS) / np.linalg.norm(residual_check_RHS)
        print(f'New residual Rphi = {residual_check}')

        if residual_check < 1e-4:
            print('Full system converged!')
            break

    # # update Np_prev and Nm_prev
    # Np_prev = Np.ravel(order='F')
    # Nm_prev = Nm.ravel(order='F')

   # ------------------------------------------------------
    # Shared colormap (defined once)
    # ------------------------------------------------------
    custom_cmap = ListedColormap(plt.cm.get_cmap('cmr.viola', 256)(np.linspace(0.2, 0.8, 256)))

    def plot_field(field, scalars_name, clim, save_path, Xint, Yint, Uint, Vint, radius=1):
        """
        Render a 2D scalar field with PyVista (chroma-keyed) and overlay
        matplotlib streamlines, then save to disk.
        """
        # --- PyVista render ---
        grid = pv.StructuredGrid(Xint, Yint, np.zeros_like(field))
        grid[scalars_name] = field.flatten(order='F')

        plotter = pv.Plotter(off_screen=True, window_size=[700, 700])
        plotter.set_background([1.0, 0.0, 1.0])  # magenta for chroma-keying
        plotter.add_mesh(
            grid,
            scalars=scalars_name,
            cmap=custom_cmap,
            clim=clim,
            show_edges=False,
            show_scalar_bar=False,
            lighting=False,
        )
        disk = pv.Disc(center=(0, -(particle_y_offset), 0.001), inner=0, outer=radius, normal=(0, 0, 1), r_res=1, c_res=100) # -(np.pi + 2)
        plotter.add_mesh(disk, color="gray", show_edges=False)
        plotter.view_xy()
        plotter.camera.tight(padding=0.0)
        pv_image = plotter.screenshot(None, return_img=True)  # (H, W, 3) uint8
        plotter.close()

        # --- Chroma-key magenta → transparent ---
        pv_rgba = np.concatenate(
            [pv_image, np.full((*pv_image.shape[:2], 1), 255, dtype=np.uint8)], axis=-1
        )
        is_background = (pv_image[:, :, 0] >= 245) & (pv_image[:, :, 1] <= 10) & (pv_image[:, :, 2] >= 245)
        pv_rgba[is_background, 3] = 0

        # --- Matplotlib composite ---
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()

        fig, ax = plt.subplots(figsize=(7, 7), dpi=100, facecolor='white')
        ax.set_facecolor('white')

        ax.imshow(
            pv_rgba,
            extent=[x_min, x_max, y_min, y_max],
            origin='upper',
            aspect='equal',
            zorder=0,
        )

        # Mask streamlines inside the disk
        U_masked = np.where(Xint**2 + (Yint + particle_y_offset)**2 <= radius**2, np.nan, Uint) # Y + 2
        V_masked = np.where(Xint**2 + (Yint + particle_y_offset)**2 <= radius**2, np.nan, Vint) # Y + 2
        ax.streamplot(Xint, Yint, U_masked, V_masked, color='black', density=2, linewidth=1, arrowsize=1.5, zorder=1)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=Normalize(vmin=clim[0], vmax=clim[1]))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=scalars_name, shrink=0.8)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()


    # # ------------------------------------------------------
    # # Plot all three fields
    # # ------------------------------------------------------
    N_net = (Np - Nm) / 2

    fields = [
        (N_net, 'N_net', [-0.5, 0.5], f'img/electrode/n_net/zhao_compare.png'),
        (Np,    'N_p',   [0,  2], f'img/electrode/n_p/zhao_compare.png'),
        (Nm,    'N_m',   [0,  2], f'img/electrode/n_m/zhao_compare.png'),
    ]

    Uint = 0.5 * (U_fluid[:-1, 1:-1] + U_fluid[1:, 1:-1])
    Vint = 0.5 * (V_fluid[1:-1, :-1] + V_fluid[1:-1, 1:])

    for field, name, clim, path in fields:
        plot_field(field, name, clim, path, Xint, Yint, Uint, Vint)

residual_check_RHS = b_Op(u_next, ctxt_BCs, U_fluid, V_fluid, electrode_p, electrode_m)
residual_check_AxOp = AxOp(u_next)
residual_check = np.linalg.norm(residual_check_AxOp - residual_check_RHS) / np.linalg.norm(residual_check_RHS)
print(f'New residual Rphi = {residual_check}')

# check Nu residual 
Nu_RHS = np.concatenate([f_bc, g_bc, h_bc, z_bc, V_bc])
lam = stokes.interleave(lam_X, lam_Y)
u_tilde = np.concatenate([U, V, P, lam, V_rigid])
K = stokes.build_K(xib, yib, [particle_x_offset, particle_y_offset], Nib)
Nu = stokes.apply_A_K(u_tilde, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x_staggered, G_y_staggered, D_x_staggered, D_y_staggered, K, cut)
residual_check_N = np.linalg.norm(Nu - Nu_RHS) / np.linalg.norm(Nu_RHS)
print(f'New residual Nu = {residual_check_N}')

# check system residual
full_system_operator_applied = np.concatenate([Nu, residual_check_AxOp])
full_system_RHS = np.concatenate([Nu_RHS, residual_check_RHS])
residual_check_full = np.linalg.norm(full_system_operator_applied - full_system_RHS) / np.linalg.norm(full_system_RHS)
print(f'Full residual = {residual_check_full}')

# # end profiling
# pr.disable()
# pr.dump_stats("profile_fast.prof")

###################################
#######  SAVE/PLOT RESULTS  #######
###################################

# Save results
savemat('zhao_compare.mat', {
    'ctxt_Rphi': ctxt,
    'u_next': u_next,
    'Xint': Xint,
    'Yint': Yint,
    'xib': xib,
    'yib': yib,
    'Phi': Phi,
    'Np': Np,
    'Nm': Nm,
    'V_rigid': V_rigid,
    'err': np.array(err),
    # 'body_x': body_interpolated_x,
    # 'body_y': body_interpolated_y,
    'U_fluid': U_fluid,
    'V_fluid': V_fluid,
    'P_fluid': P,
    'lam_X': lam_X,
    'lam_Y': lam_Y
})

# print('U asym:', np.max(np.abs(U_fluid.reshape(Ny,Nx,order='F') + np.fliplr(U_fluid.reshape(Ny,Nx,order='F')))))  # U should be ANTIsymmetric
# print('V asym:', np.max(np.abs(V_fluid.reshape(Ny,Nx,order='F') - np.fliplr(V_fluid.reshape(Ny,Nx,order='F')))))  # V should be symmetric

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xint, Yint, Phi, cmap=cmap, edgecolor='none')
ax.set_title("Phi")
ax.set_xlabel("x"); ax.set_ylabel("y")

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xint, Yint, Np, cmap=cmap, edgecolor='none')
ax.set_title("Np")
ax.set_xlabel("x"); ax.set_ylabel("y")

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xint, Yint, Nm, cmap=cmap, edgecolor='none')
ax.set_title("Nm")
ax.set_xlabel("x"); ax.set_ylabel("y")

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xint, Yint, N_net, cmap=cmap, edgecolor='none')
ax.set_title("N_net")
ax.set_xlabel("x"); ax.set_ylabel("y")

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(UGridX, UGridY, U_fluid, cmap=cmap, edgecolor='none')
ax.set_title("U_fluid")
ax.set_xlabel("x"); ax.set_ylabel("y")

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(VGridX, VGridY, V_fluid, cmap=cmap, edgecolor='none')
ax.set_title("V_fluid")
ax.set_xlabel("x"); ax.set_ylabel("y")

plt.show()
