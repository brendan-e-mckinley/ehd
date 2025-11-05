
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit
from sksparse.cholmod import cholesky
from scipy.sparse import spdiags, eye, kron, diags, lil_matrix
from scipy.sparse.linalg import spsolve, gmres, LinearOperator
from scipy.linalg import qr, lstsq
from scipy.io import loadmat, savemat
from scipy.interpolate import Akima1DInterpolator, interpn
import CPEO_utils as cpeo
import stokes_solver_utils as stokes

#######################################
#####  solve R*phi = rho(u, phi)  #####
#######################################

# Grid setup
Nx = 450  # 256; % number of grid points along one direction
L = 2.0 * np.pi 
x = np.linspace(-L/2, L/2, Nx+2) 
dx = x[1] - x[0]
y = x.copy()
dy = y[1] - y[0]

xint = x[1:-1]
yint = y[1:-1]
Ny = len(yint)

X, Y = np.meshgrid(x, y)  # make 2D grid
Xint, Yint = np.meshgrid(xint, yint)  # make 2D grid of interior points

# Make immersed boundary mats
rad = 0.25
dth = dx / rad
theta = np.arange(0, 2*np.pi - dth, dth)
Nib = len(theta)
xib = rad * np.cos(theta) 
yib = rad * np.sin(theta)
n_x = np.cos(theta)
n_y = np.sin(theta)

# Make finite difference laplacian
# Dirichlet boundary conditions
e = (1/dy**2) * np.ones(Ny)
D2_d = spdiags([e, -2*e, e], [-1, 0, 1], Ny, Ny)
I_nx = eye(Nx)
I_ny = eye(Ny)
Lap = -(kron(I_nx, D2_d) + kron(D2_d, I_ny))
dLap = cholesky(Lap) # Cholesky decomposition

# Gradients
# Dx_b = diags([np.ones(Nx), -np.ones(Nx)], offsets=[0, -1], shape=(Nx, Nx), format='lil')
# Dx_b[0, -1] = -1.0  # periodic: U[0] uses P[0] - P[Nx-1]
# D_x_backward = (Dx_b / dx).tocsr()
# G_x = kron(D_x_backward, eye(Ny, format='csr'), format='csr')

# D_y_small = diags([-np.ones(Ny), np.ones(Ny)], offsets=[0, 1], shape=(Ny, Ny), format='csr') / dy
# G_y = kron(eye(Nx, format='csr'), D_y_small, format='csr')

def periodic_diff_x(Nx, dx):
    e = np.ones(Nx)
    D = diags([-0.5*e, 0.5*e], offsets=[-1, 1], shape=(Nx, Nx), format='lil') / dx
    D[0, -1] = -0.5/dx     # wrap: f_{i-1} at i=0 -> f_{N-1}
    D[-1, 0] = 0.5/dx      # wrap: f_{i+1} at i=N-1 -> f_{0}
    return D.tocsr()

def periodic_diff_y(Ny, dy):
    e = np.ones(Ny)
    D = diags([-0.5*e, 0.5*e], offsets=[-1, 1], shape=(Ny, Ny), format='lil') / dy
    # D[0, -1] = -0.5/dy
    # D[-1, 0] = 0.5/dy
    return D.tocsr()

# Build operators
#D_x_small_full = periodic_diff_x(Nx + 2, dx)
#D_y_small_full = periodic_diff_y(Ny + 2, dy)

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

# Full 2D operator, column-major ordering
G_x_full = kron(D_x_full, eye(Ny + 2, format='csr'), format='csr')
G_y_full = kron(eye(Nx + 2, format='csr'), D_y_full,format='csr')

D_x_full_nodes = diags(
    [-0.5/dx * np.ones(Nx), 0.5/dx * np.ones(Nx)],
    offsets=[-1, 1],
    shape=(Nx, Nx + 2),
    format='csr'
)

D_y_full_nodes = diags(
    [-0.5/dy * np.ones(Ny), 0.5/dy * np.ones(Ny)],
    offsets=[-1,  1],
    shape=(Ny, Ny + 2),
    format='csr'
)

# Selector matrices that remove boundaries:
S_y = eye(Ny + 2, format='csr')[1:-1, :]   # (Ny) × (Ny+2)
S_x = eye(Nx + 2, format='csr')[1:-1, :]   # (Nx) × (Nx+2)

# 2D gradient operators (Phi_full is flattened in Fortran order)
G_x_full_nodes = kron(D_x_full_nodes, S_y, format='csr')     # (Nx*Ny) × ((Nx+2)*(Ny+2))
G_y_full_nodes = kron(S_x, D_y_full_nodes, format='csr')

# plt.spy(G_x_full_nodes)
# plt.spy(G_y_full_nodes)
# plt.show()

# Build operators
D_x_small = periodic_diff_x(Nx, dx)
D_y_small = periodic_diff_y(Ny, dy)

G_x = kron(D_x_small, eye(Ny, format='csr'), format='csr')
G_y = kron(eye(Nx, format='csr'), D_y_small, format='csr')

# Parameters
beta_BC = 7.94
sigma_bc = 0.78  # 0.68
delta_layer = 0.1  # 5*dx; %6*dx;
cut = 6 * 1.2 * dx # cutoff value

# Exact solution
def Phi_exact(x, y):
    return beta_BC * y + 0 * x

def Npm_exact(x, y):
    return 0 * x + 1.0

# Boundary conditions
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

# Compute exact solutions
Phi_BC = Phi_exact(X, Y)
N_pm_BC = Npm_exact(X, Y)

# Boundary conditions context for Schur system
ctxt_BCs_Schur = np.concatenate([
    Phi_BCs.ravel(order='F'),
    Npm_BCs.ravel(order='F'),
    Npm_BCs.ravel(order='F'),
    np.zeros(len(xib)) - (sigma_bc),
    np.zeros(len(xib)),
    np.zeros(len(xib))
])

# initial u_vec
U_fluid = np.zeros(Nx * Ny)
V_fluid = np.zeros(Nx * Ny)

# Boundary conditions context rho tilde
ctxt_BCs = np.concatenate([
    Phi_BCs.ravel(order='F'),
    Npm_BCs.ravel(order='F'),
    Npm_BCs.ravel(order='F'),
    np.zeros(len(xib)) - (sigma_bc/delta_layer),
    np.zeros(len(xib)),
    np.zeros(len(xib))
])

# Delta functions (R operator)
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
    return cpeo.Build_RHS_Schur_System(ctxt, ctxt_BCs_Schur, U_fluid, V_fluid, G_x_full_nodes, G_y_full_nodes, Lap, dLap, G_d_G, delta_layer, Nx, Ny, Nib, Jop, Jop_prime, dx)

def b_Op(ctxt, U_fluid, V_fluid):
    return cpeo.Build_RHS_rho(ctxt, ctxt_BCs, U_fluid, V_fluid, G_x_full_nodes, G_y_full_nodes, Lap, dLap, G_d_G, delta_layer, Nx, Ny, Nib, Jop, Jop_prime, dx)

def AxOp(ctxt):
    return cpeo.Constrained_Lap(ctxt, ctxt, dLap, delta_layer, Nx, Ny, Nib, Sop_prime, Jop_prime)

# Load initial conditions from .mat file
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

schurRHS = b_Op_Schur(ctxt, U_fluid, V_fluid)

# Anderson acceleration parameters
beta = 0.2
m = 50
DU = np.full((len(schurRHS), m), np.nan)
DG = np.full((len(schurRHS), m), np.nan)

tol = 1e-4
u_n = ctxt.copy()
p_guess = u_n[3*Nx*Ny:]

class GMRES_Counter:
    def __init__(self):
        self.niter = 0

    def __call__(self, xk):
        self.niter += 1

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
U, Sigma, Vh = np.linalg.svd(schurDense)

plot_singular = Sigma / np.max(Sigma)
indices = np.arange(len(plot_singular))

# Use SVD to solve
p_next = cpeo.solve_from_svd(U, Sigma, Vh, computedRHS)
G_u_n = cpeo.post_processing_compute_R(dLap, p_next, schurRHS, Nx, Ny, Nib, delta_layer, Sop_prime)

u_next = G_u_n.copy()
G_u_next = G_u_n.copy()
err = []

# Anderson acceleration loop
for its in range(100000):
    schurRHS = b_Op_Schur(u_next, U_fluid, V_fluid)
    computedRHS = cpeo.schur_rhs_R(dLap, schurRHS, Nx, Ny, Nib, delta_layer, Jop_prime)
    
    # Use SVD to solve
    p_n = cpeo.solve_from_svd(U, Sigma, Vh, computedRHS)
    G_u_next = cpeo.post_processing_compute_R(dLap, p_n, schurRHS, Nx, Ny, Nib, delta_layer, Sop_prime)

    m_n = min(m, its + 1)
    
    # Store differences
    if its < m:
        DU[:, its] = u_next - u_n
        DG[:, its] = G_u_next - G_u_n
    else:
        DU = np.roll(DU, -1, axis=1)
        DG = np.roll(DG, -1, axis=1)
        DU[:, -1] = u_next - u_n
        DG[:, -1] = G_u_next - G_u_n
    
    f_n = G_u_next - u_next
    DF = DG[:, :m_n] - DU[:, :m_n]
    
    # QR decomposition
    #Q_qr, R_qr = qr(DF, mode='economic')
    #gamma = np.linalg.solve(R_qr, Q_qr.T @ f_n)
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
    
    print(f'Iteration {its}: residual = {err_curr}')
    
    if err_curr < 1e-4:
        print('Converged!')
        break

ctxt = u_next.copy()

# verify residudal
residual_check_RHS = b_Op(u_next, U_fluid, V_fluid)
residual_check_AxOp = AxOp(u_next)
residual_check = np.linalg.norm(residual_check_AxOp - residual_check_RHS) / np.linalg.norm(residual_check_RHS)

# compute body forces 
Phi_full = np.zeros((Ny+2, Nx+2))
Np_full = np.ones((Ny+2, Nx+2))
Nm_full = np.ones((Ny+2, Nx+2))

Phi_full[1:-1, 1:-1] = Phi
Phi_full[1:-1, 0] = Phi[:, -1]
Phi_full[1:-1, -1] = Phi[:, -1]
Phi_full[0, :] = -25
Phi_full[-1, :] = 25
Np_full[1:-1, 1:-1] = Np
Nm_full[1:-1, 1:-1] = Nm

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Phi_full, cmap=cmap, edgecolor='none')
ax.set_title("Phi")
ax.set_xlabel("x"); ax.set_ylabel("y")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Np_full, cmap=cmap, edgecolor='none')
ax.set_title("N+")
ax.set_xlabel("x"); ax.set_ylabel("y")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Nm_full, cmap=cmap, edgecolor='none')
ax.set_title("N-")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.show()

Grad_x_Phi_Flat = (G_x_full @ Phi_full.ravel(order='F'))
Grad_y_Phi_Flat = (G_y_full @ Phi_full.ravel(order='F'))

Np_relevant_y = Np_full[1:-1,:]
Nm_relevant_y = Nm_full[1:-1,:]
Np_relevant_x = Np_full[:,1:-1]
Nm_relevant_x = Nm_full[:,1:-1]

bodyForces_x = (Np_relevant_x.ravel(order='F') - Nm_relevant_x.ravel(order='F')) / (2 * delta_layer**2) * Grad_x_Phi_Flat
bodyForces_y = (Np_relevant_y.ravel(order='F') - Nm_relevant_y.ravel(order='F')) / (2 * delta_layer**2) * Grad_y_Phi_Flat

body_x = bodyForces_x.reshape(Ny + 2, Nx, order='F')
body_y = bodyForces_y.reshape(Ny, Nx + 2, order='F')
body_interpolated_x = 0.5 * (body_x[:-1, :] + body_x[1:, :])
body_interpolated_y = 0.5 * (body_y[:, :-1] + body_y[:, 1:])

# --- Corners (average of nearest ghost neighbors) ---
# body_full[0, 0]     = 0.5 * (body_full[0, 1] + body_full[1, 0])
# body_full[0, -1]    = 0.5 * (body_full[0, -2] + body_full[1, -1])
# body_full[-1, 0]    = 0.5 * (body_full[-2, 0] + body_full[-1, 1])
# body_full[-1, -1]   = 0.5 * (body_full[-1, -2] + body_full[-2, -1])

################################
#####  solve N*u = F(phi)  #####
################################

# Parameters
x = np.linspace(-L/2, L/2, Nx + 2)
dx = x[1] - x[0]
dx2 = dx**2
y = x.copy()
dy = dx

#tol = 0.01 * dx**2

cut = 6 * 1.2 * dx # cutoff value

delta_x, delta_y = stokes.make_composite_deltas(dx, n=3)

x_trunc = x[1:-1]    # length Nx
y_trunc = y[1:-1]    # length Ny
x_mid = x + dx / 2
y_mid = y + dy / 2
x_offset = x_mid[:-1]
y_offset = y_mid[:-1]

UGridX, UGridY = np.meshgrid(x_trunc, y_offset)
VGridX, VGridY = np.meshgrid(x_offset, y_trunc)

N_U = Nx * (Ny + 1)
N_V = (Nx + 1) * Ny
N_P = (Nx + 1) * (Ny + 1)

# BCs
V_lower = 0
V_upper = 0

# RHS
f = body_interpolated_x.ravel(order='F')
g = body_interpolated_y.ravel(order='F')
h_bc = np.zeros((Ny + 1, Nx + 1)).ravel(order='F')
z_x = np.zeros(Nib)
z_y = np.zeros(Nib)

f_bc_mat = np.zeros((Ny + 1, Nx))
g_bc_mat = np.zeros((Ny, Nx + 1))

f_bc = f.ravel(order='F') + f_bc_mat.ravel(order='F')
g_bc = g.ravel(order='F') + g_bc_mat.ravel(order='F')

Lap_U, Lap_V = stokes.build_staggered_Laps(Nx, dx)
G_x_staggered, G_y_staggered, D_x_staggered, D_y_staggered = stokes.build_staggered_Grads_Divs(Nx, dx)

U, V, P, lam_X, lam_Y = stokes.solve(-L/2, L/2, f_bc, g_bc, h_bc, z_x, z_y, rad, Nx, tol)

# # check Nu residual 
# Nu_RHS = np.concatenate([f_bc, g_bc, h_bc, z_x, z_y])
# u_tilde = np.concatenate([U, V, P, lam_X, lam_Y])
# Nu = stokes.apply_A(u_tilde, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x_staggered, G_y_staggered, D_x_staggered, D_y_staggered)
# residual_check_N = np.linalg.norm(Nu - Nu_RHS) / np.linalg.norm(Nu_RHS)
# print(f'residual Nu = {residual_check_N}')

# # figure out where residual is worst
# offset = 0
# residual_check_U = np.linalg.norm(Nu[:N_U] - Nu_RHS[:N_U]) / np.linalg.norm(Nu_RHS[:N_U])
# for i in range(Nx):
#     relevantNu = Nu[i*Nx:i*Nx+Nx]
#     relevantRHS = Nu_RHS[i*Nx:i*Nx+Nx]
#     residual_check = np.linalg.norm(relevantNu - relevantRHS) / np.linalg.norm(relevantRHS)
#     if residual_check > 1e-5:
#         print(f'high residual: {residual_check}')
# offset += N_U
# residual_check_V = np.linalg.norm(Nu[offset:offset + N_V] - Nu_RHS[offset:offset + N_V]) / np.linalg.norm(Nu_RHS[offset:offset + N_V])
# offset += N_V
# residual_check_P = np.linalg.norm(Nu[offset:offset + N_P] - Nu_RHS[offset:offset + N_P])
# offset += N_P
# residual_check_lam_X = np.linalg.norm(Nu[offset:offset + Nib] - Nu_RHS[offset:offset + Nib])
# offset += Nib
# residual_check_lam_Y = np.linalg.norm(Nu[offset:offset + Nib] - Nu_RHS[offset:offset + Nib])


#P = P - np.mean(P)

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

# Build full arrays with ghost rows/cols similar to MATLAB, but Dirichlet zero ghosts
UFull = np.zeros((Ny + 2, Nx + 2))
UFull[1:Ny + 1, 1:Nx + 1] = U_interpolated

VFull = np.zeros((Ny + 2, Nx + 2))
VFull[1:Ny + 1, 1:Nx + 1] = V_interpolated

xplot = np.linspace(-L/2, L/2, Nx + 2)
yplot = np.linspace(-L/2, L/2, Ny + 2)
Xplot, Yplot = np.meshgrid(xplot, yplot)

# # Define circular mask (radius 0.25 centered at origin)
# radius = 0.25
# mask = (Xplot**2 + Yplot**2) <= radius**2

# # Apply mask to UFull and VFull
# UFull[mask] = np.nan
# VFull[mask] = np.nan

# Plotting (3D surface)
cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xplot, Yplot, UFull, cmap=cmap, edgecolor='none')
ax.set_title("U")
ax.set_xlabel("x"); ax.set_ylabel("y")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xplot, Yplot, VFull, cmap=cmap, edgecolor='none')
ax.set_title("V")
ax.set_xlabel("x"); ax.set_ylabel("y")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(Xplot, Yplot, PFull, cmap=cmap, edgecolor='none')
# ax.set_title("P")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(Xplot, Yplot, np.sqrt(UFull**2 + VFull**2), cmap=cmap, edgecolor='none')
# ax.set_title("|velocity|")
# ax.set_xlabel("x"); ax.set_ylabel("y")

plt.figure(figsize=(8, 6))
plt.streamplot(Xplot, Yplot, UFull, VFull, color='red', density=10, linewidth=1, arrowsize=1.5)
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Flow Field Streamline Plot')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(True)
plt.show()

U_fluid = (UFull[1:Ny+1,1:Nx+1]).ravel(order='F')
V_fluid = (VFull[1:Ny+1,1:Nx+1]).ravel(order='F')

# full system loop
for its in range(100000):
    # solve Rphi = rho with computed U and V

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
    U, Sigma, Vh = np.linalg.svd(schurDense)

    plot_singular = Sigma / np.max(Sigma)
    indices = np.arange(len(plot_singular))

    # Use SVD to solve
    p_next = cpeo.solve_from_svd(U, Sigma, Vh, computedRHS)
    G_u_n = cpeo.post_processing_compute_R(dLap, p_next, schurRHS, Nx, Ny, Nib, delta_layer, Sop_prime)

    u_next = G_u_n.copy()
    G_u_next = G_u_n.copy()
    err = []

    # Anderson acceleration loop
    for inner_its in range(100000):
        schurRHS = b_Op_Schur(u_next, U_fluid, V_fluid)
        computedRHS = cpeo.schur_rhs_R(dLap, schurRHS, Nx, Ny, Nib, delta_layer, Jop_prime)
        
        # Use SVD to solve
        p_n = cpeo.solve_from_svd(U, Sigma, Vh, computedRHS)
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
        #Q_qr, R_qr = qr(DF, mode='economic')
        #gamma = np.linalg.solve(R_qr, Q_qr.T @ f_n)
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

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    cmap = plt.cm.spring
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Phi_full, cmap=cmap, edgecolor='none')
    ax.set_title("Phi")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Np_full, cmap=cmap, edgecolor='none')
    ax.set_title("N+")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Nm_full, cmap=cmap, edgecolor='none')
    ax.set_title("N-")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    Grad_x_Phi_Flat = (G_x_full @ Phi_full.ravel(order='F'))
    Grad_y_Phi_Flat = (G_y_full @ Phi_full.ravel(order='F'))

    Np_relevant_y = Np_full[1:-1,:]
    Nm_relevant_y = Nm_full[1:-1,:]
    Np_relevant_x = Np_full[:,1:-1]
    Nm_relevant_x = Nm_full[:,1:-1]

    bodyForces_x = (Np_relevant_x.ravel(order='F') - Nm_relevant_x.ravel(order='F')) / (2 * delta_layer**2) * Grad_x_Phi_Flat
    bodyForces_y = (Np_relevant_y.ravel(order='F') - Nm_relevant_y.ravel(order='F')) / (2 * delta_layer**2) * Grad_y_Phi_Flat

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

    U, V, P, lam_X, lam_Y = stokes.solve(-L/2, L/2, f_bc, g_bc, h_bc, z_x, z_y, rad, Nx, tol)

    #P = P - np.mean(P)
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

    # Build full arrays with ghost rows/cols similar to MATLAB, but Dirichlet zero ghosts
    UFull = np.zeros((Ny + 2, Nx + 2))
    UFull[1:Ny + 1, 1:Nx + 1] = U_interpolated

    VFull = np.zeros((Ny + 2, Nx + 2))
    VFull[1:Ny + 1, 1:Nx + 1] = V_interpolated

    U_fluid = (UFull[1:Ny+1,1:Nx+1]).ravel(order='F')
    V_fluid = (VFull[1:Ny+1,1:Nx+1]).ravel(order='F')

    residual_check_RHS = b_Op(u_next, U_fluid, V_fluid)
    residual_check_AxOp = AxOp(u_next)
    residual_check = np.linalg.norm(residual_check_AxOp - residual_check_RHS) / np.linalg.norm(residual_check_RHS)
    print(f'New residual = {residual_check}')

    if residual_check < 1e-4:
        print('Full system converged!')
        break

# check Nu residual 
Nu_RHS = np.concatenate([f_bc, g_bc, h_bc, z_x, z_y])
u_tilde = np.concatenate([U, V, P, lam_X, lam_Y])
Nu = stokes.apply_A(u_tilde, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x_staggered, G_y_staggered, D_x_staggered, D_y_staggered)
residual_check_N = np.linalg.norm(Nu - Nu_RHS) / np.linalg.norm(Nu_RHS)
print(f'New residual Nu = {residual_check_N}')

# check system residual
full_system_operator_applied = np.concatenate([Nu, residual_check_AxOp])
full_system_RHS = np.concatenate([Nu_RHS, residual_check_RHS])
residual_check_full = np.linalg.norm(full_system_operator_applied - full_system_RHS) / np.linalg.norm(full_system_RHS)
print(f'Full residual = {residual_check_full}')

# figure out where residual is worst
# offset = 0
# residual_check_U = np.linalg.norm(Nu[:N_U] - RHS[:N_U]) / np.linalg.norm(RHS[:N_U])
# offset += N_U
# residual_check_V = np.linalg.norm(Nu[offset:offset + N_V] - RHS[offset:offset + N_V]) / np.linalg.norm(RHS[offset:offset + N_V])
# offset += N_V
# residual_check_P = np.linalg.norm(Nu[offset:offset + N_P] - RHS[offset:offset + N_P]) / np.linalg.norm(RHS[offset:offset + N_P])
# offset += N_P
# residual_check_lam_X = np.linalg.norm(Nu[offset:offset + Nib] - RHS[offset:offset + Nib]) / np.linalg.norm(RHS[offset:offset + Nib])
# offset += Nib
# residual_check_lam_Y = np.linalg.norm(Nu[offset:offset + Nib] - RHS[offset:offset + Nib]) / np.linalg.norm(RHS[offset:offset + Nib])

# Define circular mask (radius 0.25 centered at origin)
# radius = 0.25
# mask = (Xplot**2 + Yplot**2) <= radius**2

# # Apply mask to UFull and VFull
# UFull[mask] = np.nan
# VFull[mask] = np.nan

plt.figure(figsize=(8, 6))
plt.streamplot(Xplot, Yplot, UFull, VFull, color='red', density=10, linewidth=1, arrowsize=1.5)
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Flow Field Streamline Plot')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(True)

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xplot, Yplot, UFull, cmap=cmap, edgecolor='none')
ax.set_title("U")
ax.set_xlabel("x"); ax.set_ylabel("y")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xplot, Yplot, VFull, cmap=cmap, edgecolor='none')
ax.set_title("V")
ax.set_xlabel("x"); ax.set_ylabel("y")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xplot, Yplot, np.sqrt(UFull**2 + VFull**2), cmap=cmap, edgecolor='none')
ax.set_title("|velocity|")
ax.set_xlabel("x"); ax.set_ylabel("y")

plt.show()
