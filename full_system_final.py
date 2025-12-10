
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
import CPEO_utils as cpeo
import stokes_solver_utils_fast as stokes

###########################
######  PARAMETERS  #######
###########################

## Grid parameters
Nx = 450  # 256; % number of grid points along one direction
L = 2.0 * np.pi 
x = np.linspace(-L/2, L/2, Nx+2) 
dx = x[1] - x[0]
y = x.copy()
dy = y[1] - y[0]

## Miscellaneous parameters
tol = 1e-4
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
Ny = len(yint)

X, Y = np.meshgrid(x, y)
Xint, Yint = np.meshgrid(xint, yint)

## Staggered grid (horizontal faces for V, vertical faces for U, cell centers for P)
N_U = Nx * (Ny + 1)
N_V = (Nx + 1) * Ny
N_P = (Nx + 1) * (Ny + 1)
x_trunc = x[1:-1]    # length Nx
y_trunc = y[1:-1]    # length Ny
x_mid = x + dx / 2
y_mid = y + dy / 2
x_offset = x_mid[:-1]
y_offset = y_mid[:-1]

UGridX, UGridY = np.meshgrid(x_trunc, y_offset)
VGridX, VGridY = np.meshgrid(x_offset, y_trunc)

## Immersed boundary
rad = 0.25
dth = dx / rad
theta = np.arange(0, 2*np.pi - dth, dth)
Nib = len(theta)
xib = rad * np.cos(theta) 
yib = rad * np.sin(theta)
n_x = np.cos(theta)
n_y = np.sin(theta)

#########################
######  OPERATORS  ######
#########################

## Laplacian operators
# Staggered laplacians (dirichlet boundary conditions in y, dirichlet boundary conditions in x)
Lap_U, Lap_V = stokes.build_staggered_Laps(Nx, dx)

# Nodal laplacian (dirichlet boundary conditions in y, periodic in x)
e = (1/dy**2) * np.ones(Ny)
D2_d = spdiags([e, -2*e, e], [-1, 0, 1], Ny, Ny)
I_nx = eye(Nx)
I_ny = eye(Ny)
Lap = -(kron(I_nx, D2_d) + kron(D2_d, I_ny))
dLap = cholesky(Lap) # Cholesky decomposition

## Gradient/divergence operators
def periodic_centered_diff_x(Nx, dx):
    e = np.ones(Nx)
    D = diags([-0.5*e, 0.5*e], offsets=[-1, 1], shape=(Nx, Nx), format='lil') / dx
    D[0, -1] = -0.5/dx     # wrap: f_{i-1} at i=0 -> f_{N-1}
    D[-1, 0] = 0.5/dx      # wrap: f_{i+1} at i=N-1 -> f_{0}
    return D.tocsr()

def centered_diff_y(Ny, dy):
    e = np.ones(Ny)
    D = diags([-0.5*e, 0.5*e], offsets=[-1, 1], shape=(Ny, Ny), format='lil') / dy
    return D.tocsr()

# Full gradient operators ((Nx + 2) * (Ny + 2)) including boundaries
D_x_full = diags(
    [-0.5/dx * np.ones(Nx), 0.5/dx * np.ones(Nx)],
    offsets=[-1, 1],
    shape=(Nx, Nx + 2)
).tocsr()
D_y_full = diags(
    [-0.5/dy * np.ones(Ny),  0.5/dy * np.ones(Ny)],
    offsets=[-1,  1],
    shape=(Ny, Ny + 2),
    format='csr'
)
G_x_full = kron(D_x_full, eye(Ny + 2, format='csr'), format='csr')
G_y_full = kron(eye(Nx + 2, format='csr'), D_y_full,format='csr')

# Nodal gradient operators (internal points only, excludes boundaries)
D_x_nodes = diags(
    [-0.5/dx * np.ones(Nx), 0.5/dx * np.ones(Nx)],
    offsets=[-1, 1],
    shape=(Nx, Nx + 2),
    format='csr'
)

D_y_nodes = diags(
    [-0.5/dy * np.ones(Ny), 0.5/dy * np.ones(Ny)],
    offsets=[-1,  1],
    shape=(Ny, Ny + 2),
    format='csr'
)
S_y = eye(Ny + 2, format='csr')[1:-1, :]   # (Ny) × (Ny+2)
S_x = eye(Nx + 2, format='csr')[1:-1, :]   # (Nx) × (Nx+2)
G_x_nodes = kron(D_x_nodes, S_y, format='csr')     # (Nx*Ny) × ((Nx+2)*(Ny+2))
G_y_nodes = kron(S_x, D_y_nodes, format='csr')     # (Nx*Ny) × ((Nx+2)*(Ny+2))

# Staggered gradient and divergence operators
G_x_staggered, G_y_staggered, D_x_staggered, D_y_staggered = stokes.build_staggered_Grads_Divs(Nx, dx)

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
    return beta_BC * y + 0 * x
def Npm_exact(x, y):
    return 0 * x + 1.0

# Compute exact solutions
Phi_BC = Phi_exact(X, Y)
N_pm_BC = Npm_exact(X, Y)

## Boundary conditions for Rphi = rho system
Phi_BCs = np.zeros_like(Xint)
Npm_BCs = np.zeros_like(Xint)

Phi_BCs[0, :] = (1/dy/dy) * Phi_exact(xint, Y[0, 0])
Phi_BCs[-1, :] += (1/dy/dy) * Phi_exact(xint, Y[-1, -1])
Phi_BCs[:, 0] += (1/dx/dx) * Phi_exact(X[0, 0], yint)
Phi_BCs[:, -1] += (1/dx/dx) * Phi_exact(X[-1, -1], yint)

Npm_BCs[0, :] = (1/dy/dy) * Npm_exact(xint, Y[0, 0])
Npm_BCs[-1, :] += (1/dy/dy) * Npm_exact(xint, Y[-1, -1])
Npm_BCs[:, 0] += (1/dx/dx) * Npm_exact(X[0, 0], yint)
Npm_BCs[:, -1] += (1/dx/dx) * Npm_exact(X[-1, -1], yint)

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

## Initial conditions for Rphi = rho system
ld = loadmat('BC_run_N_300_r0p25.mat')
METHOD = 'cubic'  # equivalent to 'makima' in MATLAB

Ny_ld = int(ld['Ny'][0, 0])
Nx_ld = int(ld['Nx'][0, 0])
Nib_ld = int(ld['Nib'][0, 0])
sz = Ny_ld * Nx_ld

ctxt_ld = ld['ctxt'].ravel(order='F')
Phi_ld = ctxt_ld[:sz].reshape(Ny_ld, Nx_ld, order='F')
N_p_ld = ctxt_ld[sz:2*sz].reshape(Ny_ld, Nx_ld, order='F')
N_m_ld = ctxt_ld[2*sz:3*sz].reshape(Ny_ld, Nx_ld, order='F')
Q_ld = ctxt_ld[3*sz:3*sz+Nib_ld]
Q_p_ld = ctxt_ld[3*sz+Nib_ld:3*sz+2*Nib_ld]
Q_m_ld = ctxt_ld[3*sz+2*Nib_ld:3*sz+3*Nib_ld]

Xint_ld = ld['Xint']
Yint_ld = ld['Yint']
theta_ld = ld['theta'].ravel()

# Extract the coordinate vectors from the loaded grid
x_ld = Xint_ld[0, :]  # First row gives x-coordinates
y_ld = Yint_ld[:, 0]  # First column gives y-coordinates

# Interpolate initial guesses
Phi_init = interpn((x_ld, y_ld), Phi_ld, (Xint.T, Yint.T), method='linear', bounds_error=False, fill_value=None)
N_p_init = interpn((x_ld, y_ld), N_p_ld, (Xint.T, Yint.T), method='nearest', bounds_error=False, fill_value=None)
N_m_init = interpn((x_ld, y_ld), N_m_ld, (Xint.T, Yint.T), method='nearest', bounds_error=False, fill_value=None)
Q_init = Akima1DInterpolator(theta_ld, Q_ld, method="makima", extrapolate=True)(theta)
Q_p_init = Akima1DInterpolator(theta_ld, Q_p_ld, method="makima", extrapolate=True)(theta)
Q_m_init = Akima1DInterpolator(theta_ld, Q_m_ld, method="makima", extrapolate=True)(theta)

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

## Boundary conditions for Nu = F system
f_bc_mat = np.zeros((Ny + 1, Nx))
g_bc_mat = np.zeros((Ny, Nx + 1))
h_bc = np.zeros((Ny + 1, Nx + 1)).ravel(order='F')
z_x = np.zeros(Nib)
z_y = np.zeros(Nib)

## Initial conditions for Nu = F system
U_fluid = np.zeros(Nx * Ny)
V_fluid = np.zeros(Nx * Ny)

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

def G_d_G(Phi, N_pm):
    return cpeo.Grad_dot_Grad(Phi, N_pm, dx, dy, Nx, Ny, Phi_BC, N_pm_BC)

def b_Op_Schur(ctxt, U_fluid, V_fluid):
    return cpeo.Build_RHS_Schur_System(ctxt, ctxt_BCs_Schur, U_fluid, V_fluid, G_x_nodes, G_y_nodes, Lap, dLap, G_d_G, delta_layer, Nx, Ny, Nib, Jop, Jop_prime, dx)

def b_Op(ctxt, U_fluid, V_fluid):
    return cpeo.Build_RHS_rho(ctxt, ctxt_BCs, U_fluid, V_fluid, G_x_nodes, G_y_nodes, Lap, dLap, G_d_G, delta_layer, Nx, Ny, Nib, Jop, Jop_prime, dx)

def AxOp(ctxt):
    return cpeo.Constrained_Lap(ctxt, ctxt, dLap, delta_layer, Nx, Ny, Nib, Sop_prime, Jop_prime)

delta_x, delta_y = stokes.make_composite_deltas(dx, n=3)

##########################
#####  SOLVER SETUP  #####
##########################

## Initialize variables for Rphi = rho solve
schurRHS = b_Op_Schur(ctxt, U_fluid, V_fluid)
DU = np.full((len(schurRHS), m), np.nan)
DG = np.full((len(schurRHS), m), np.nan)

u_n = ctxt.copy()
p_guess = u_n[3*Nx*Ny:]

# Build dense Schur matrix
schurOp = cpeo.SchurLinearOperator_R(dLap, Nib*3, Nib, Nx, Ny, delta_layer, Sop_prime, Jop_prime)
schurRHS = b_Op_Schur(ctxt, U_fluid, V_fluid)
computedRHS = cpeo.schur_rhs_R(dLap, schurRHS, Nx, Ny, Nib, delta_layer, Jop_prime)
schurDense = np.zeros((Nib * 3, Nib * 3))
for col in range(Nib * 3): 
    eye_mat = np.zeros(Nib * 3)
    eye_mat[col] = 1
    p = eye_mat[0:Nib]
    p_p = eye_mat[Nib:2*Nib]
    p_m = eye_mat[2*Nib:3*Nib]
    res_1, res_2, res_3 = cpeo.apply_Schur_R(dLap, [p, p_p, p_m], delta_layer, Nx, Ny, Sop_prime, Jop_prime)
    schurDense[:,col] = np.concatenate([res_1, res_2, res_3])

# Compute SVD once
U_schur, Sigma_schur, Vh_schur = np.linalg.svd(schurDense)

# Use SVD solve to initialize 
p_next = cpeo.solve_from_svd(U_schur, Sigma_schur, Vh_schur, computedRHS)
G_u_n = cpeo.post_processing_compute_R(dLap, p_next, schurRHS, Nx, Ny, Nib, delta_layer, Sop_prime)
u_next = G_u_n.copy()
G_u_next = G_u_n.copy()
err = []

# compute body forces 
Phi_full = np.zeros((Ny+2, Nx+2))
Np_full = np.ones((Ny+2, Nx+2))
Nm_full = np.ones((Ny+2, Nx+2))

Phi_full[1:-1, 1:-1] = Phi
Phi_full[1:-1, 0] = Phi[:, -1]
Phi_full[1:-1, -1] = Phi[:, -1]
Phi_full[0, :] = -25 # TODO: HACKY AND WRONG, FIX THIS
Phi_full[-1, :] = 25 # TODO: HACKY AND WRONG, FIX THIS
Np_full[1:-1, 1:-1] = Np
Nm_full[1:-1, 1:-1] = Nm

Grad_x_Phi_Flat = (G_x_full @ Phi_full.ravel(order='F'))
Grad_y_Phi_Flat = (G_y_full @ Phi_full.ravel(order='F'))

Np_relevant_y = Np_full[1:-1,:]
Nm_relevant_y = Nm_full[1:-1,:]
Np_relevant_x = Np_full[:,1:-1]
Nm_relevant_x = Nm_full[:,1:-1]

bodyForces_x = -(Np_relevant_x.ravel(order='F') - Nm_relevant_x.ravel(order='F')) / (2 * delta_layer**2) * Grad_x_Phi_Flat
bodyForces_y = -(Np_relevant_y.ravel(order='F') - Nm_relevant_y.ravel(order='F')) / (2 * delta_layer**2) * Grad_y_Phi_Flat

body_x = bodyForces_x.reshape(Ny + 2, Nx, order='F')
body_y = bodyForces_y.reshape(Ny, Nx + 2, order='F')
body_interpolated_x = 0.5 * (body_x[:-1, :] + body_x[1:, :])
body_interpolated_y = 0.5 * (body_y[:, :-1] + body_y[:, 1:])

## Initialize variables for Nu = F solve
f = body_interpolated_x.ravel(order='F')
g = body_interpolated_y.ravel(order='F')

f_bc = f.ravel(order='F') + f_bc_mat.ravel(order='F')
g_bc = g.ravel(order='F') + g_bc_mat.ravel(order='F')

# Build dense Schur matrix for hydrodynamic system
schurDense_N = np.zeros((Nib * 2, Nib * 2))
for col in range(Nib * 2): 
    eye_mat_N = np.zeros(Nib * 2)
    eye_mat_N[col] = 1
    lam_X = eye_mat_N[0:Nib]
    lam_Y = eye_mat_N[Nib:]
    schurDense_N[:,col] = stokes.apply_Schur_new(lam_X, lam_Y, stokes_LU, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, cut)

U_schur_fluid, Sigma_schur_fluid, Vh_schur_fluid = np.linalg.svd(schurDense_N)

U, V, P, lam_X, lam_Y = stokes.solve_factorized(-L/2, L/2, stokes_LU, U_schur_fluid, Sigma_schur_fluid, Vh_schur_fluid, f_bc, g_bc, h_bc, z_x, z_y, rad, Nx, tol, cut)
#U, V, P, lam_X, lam_Y = stokes.solve(-L/2, L/2, f_bc, g_bc, h_bc, z_x, z_y, rad, Nx, tol, cut)

Uplot = U.reshape((Ny + 1, Nx), order='F')
Vplot = V.reshape((Ny, Nx + 1), order='F')
Pplot = P.reshape((Ny + 1, Nx + 1), order='F')

U_interpolated = np.zeros([Ny, Nx])
V_interpolated = np.zeros([Ny, Nx])
for col_index in range(Uplot.shape[1]):
    col = Uplot[:, col_index]
    for row_index in range(len(col) - 1):
        midpoint = (col[row_index] + col[row_index+1]) / 2
        U_interpolated[row_index, col_index] = midpoint


for row_index in range(Vplot.shape[0]):
    row = Vplot[row_index, :]
    for col_index in range(len(row) - 1):
        midpoint = (row[col_index] + row[col_index+1]) / 2
        V_interpolated[row_index, col_index] = midpoint

UFull = np.zeros((Ny + 2, Nx + 2))
UFull[1:Ny + 1, 1:Nx + 1] = U_interpolated

VFull = np.zeros((Ny + 2, Nx + 2))
VFull[1:Ny + 1, 1:Nx + 1] = V_interpolated

# Get initial fluid velocity values
U_fluid = (UFull[1:Ny+1,1:Nx+1]).ravel(order='F')
V_fluid = (VFull[1:Ny+1,1:Nx+1]).ravel(order='F')

## Begin profiling here
pr = cProfile.Profile()
pr.enable()

##################################
#######  FULL SYSTEM LOOP  #######
##################################
for its in range(100000):
    #######################################
    #####  solve R*phi = Rho(u, phi)  #####
    #######################################

    schurRHS = b_Op_Schur(ctxt, U_fluid, V_fluid)
    computedRHS = cpeo.schur_rhs_R(dLap, schurRHS, Nx, Ny, Nib, delta_layer, Jop_prime)

    # Use SVD to solve
    p_next = cpeo.solve_from_svd(U_schur, Sigma_schur, Vh_schur, computedRHS)
    G_u_n = cpeo.post_processing_compute_R(dLap, p_next, schurRHS, Nx, Ny, Nib, delta_layer, Sop_prime)

    u_next = G_u_n.copy()
    G_u_next = G_u_n.copy()
    err = []

    # Anderson acceleration loop
    for inner_its in range(100000):
        schurRHS = b_Op_Schur(u_next, U_fluid, V_fluid)
        computedRHS = cpeo.schur_rhs_R(dLap, schurRHS, Nx, Ny, Nib, delta_layer, Jop_prime)
        
        # Use SVD to solve
        p_n = cpeo.solve_from_svd(U_schur, Sigma_schur, Vh_schur, computedRHS)
        G_u_next = cpeo.post_processing_compute_R(dLap, p_n, schurRHS, Nx, Ny, Nib, delta_layer, Sop_prime)

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
        res1, res2, res3 = cpeo.apply_Schur_R(dLap, [p, p_p, p_m], delta_layer, Nx, Ny, Sop_prime, Jop_prime)
        schur_next = np.concatenate([res1, res2, res3])
        err_curr = np.linalg.norm(schur_next - computedRHS) / np.linalg.norm(computedRHS)
        
        err.append(err_curr)
        
        print(f'Iteration {inner_its}: residual = {err_curr}')
        
        if err_curr < 1e-4:
            print('Rphi = rho Converged!')
            break
    
    ctxt = u_next.copy()
    
    # compute body forces 
    Phi_full = np.zeros((Ny+2, Nx+2))
    Np_full = np.ones((Ny+2, Nx+2))
    Nm_full = np.ones((Ny+2, Nx+2))

    Phi_full[1:-1, 1:-1] = Phi
    Phi_full[1:-1, 0] = Phi[:, -1]
    Phi_full[1:-1, -1] = Phi[:, -1]
    Phi_full[0, 1:-1] = -25
    Phi_full[-1, 1:-1] = 25
    Np_full[1:-1, 1:-1] = Np
    Nm_full[1:-1, 1:-1] = Nm

    Grad_x_Phi_Flat = (G_x_full @ Phi_full.ravel(order='F'))
    Grad_y_Phi_Flat = (G_y_full @ Phi_full.ravel(order='F'))

    Np_relevant_y = Np_full[1:-1,:]
    Nm_relevant_y = Nm_full[1:-1,:]
    Np_relevant_x = Np_full[:,1:-1]
    Nm_relevant_x = Nm_full[:,1:-1]

    bodyForces_x = -(Np_relevant_x.ravel(order='F') - Nm_relevant_x.ravel(order='F')) / (2 * delta_layer**2) * Grad_x_Phi_Flat
    bodyForces_y = -(Np_relevant_y.ravel(order='F') - Nm_relevant_y.ravel(order='F')) / (2 * delta_layer**2) * Grad_y_Phi_Flat

    body_x = bodyForces_x.reshape(Ny + 2, Nx, order='F')
    body_y = bodyForces_y.reshape(Ny, Nx + 2, order='F')
    body_interpolated_x = 0.5 * (body_x[:-1, :] + body_x[1:, :])
    body_interpolated_y = 0.5 * (body_y[:, :-1] + body_y[:, 1:])

    ################################
    #####  solve N*u = F(phi)  #####
    ################################

    # RHS
    f = body_interpolated_x.ravel(order='F')
    g = body_interpolated_y.ravel(order='F')

    f_bc = f.ravel(order='F') + f_bc_mat.ravel(order='F')
    g_bc = g.ravel(order='F') + g_bc_mat.ravel(order='F')

    U, V, P, lam_X, lam_Y = stokes.solve_factorized(-L/2, L/2, stokes_LU, U_schur_fluid, Sigma_schur_fluid, Vh_schur_fluid, f_bc, g_bc, h_bc, z_x, z_y, rad, Nx, tol, cut)
    #U, V, P, lam_X, lam_Y = stokes.solve(-L/2, L/2, f_bc, g_bc, h_bc, z_x, z_y, rad, Nx, tol, cut)

    Uplot = U.reshape((Ny + 1, Nx), order='F')
    Vplot = V.reshape((Ny, Nx + 1), order='F')
    Pplot = P.reshape((Ny + 1, Nx + 1), order='F')

    U_interpolated = np.zeros([Ny, Nx])
    V_interpolated = np.zeros([Ny, Nx])
    for col_index in range(Uplot.shape[1]):
        col = Uplot[:, col_index]
        for row_index in range(len(col) - 1):
            midpoint = (col[row_index] + col[row_index+1]) / 2
            U_interpolated[row_index, col_index] = midpoint

    for row_index in range(Vplot.shape[0]):
        row = Vplot[row_index, :]
        for col_index in range(len(row) - 1):
            midpoint = (row[col_index] + row[col_index+1]) / 2
            V_interpolated[row_index, col_index] = midpoint

    UFull = np.zeros((Ny + 2, Nx + 2))
    UFull[1:Ny + 1, 1:Nx + 1] = U_interpolated

    VFull = np.zeros((Ny + 2, Nx + 2))
    VFull[1:Ny + 1, 1:Nx + 1] = V_interpolated

    U_fluid = (UFull[1:Ny+1,1:Nx+1]).ravel(order='F')
    V_fluid = (VFull[1:Ny+1,1:Nx+1]).ravel(order='F')

    residual_check_RHS = b_Op(u_next, U_fluid, V_fluid)
    residual_check_AxOp = AxOp(u_next)
    residual_check = np.linalg.norm(residual_check_AxOp - residual_check_RHS) / np.linalg.norm(residual_check_RHS)
    print(f'New residual Rphi = {residual_check}')

    if residual_check < 1e-4:
        print('Full system converged!')
        break

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
