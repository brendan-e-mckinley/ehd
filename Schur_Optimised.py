import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit
from sksparse.cholmod import cholesky
from scipy.sparse import spdiags, eye, kron
from scipy.sparse.linalg import spsolve, gmres, LinearOperator, splu
from scipy.linalg import qr
from scipy.io import loadmat, savemat
from scipy.interpolate import RegularGridInterpolator, Akima1DInterpolator, interpn

t = time.time()

# Plotting parameters
plt.rcParams.update({
    'font.size': 35,
    'lines.linewidth': 3,
    'axes.labelsize': 35,
    'xtick.labelsize': 35,
    'ytick.labelsize': 35
})

# Grid setup
Nx = 450  # 256; % number of grid points along one direction
plt.clf()
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

# Make finite difference laplacian
# Dirichlet boundary conditions
e = (1/dy**2) * np.ones(Ny)
D2_d = spdiags([e, -2*e, e], [-1, 0, 1], Ny, Ny)
I_nx = eye(Nx)
I_ny = eye(Ny)
Lap = -(kron(I_nx, D2_d) + kron(D2_d, I_ny))
dLap = cholesky(Lap) # Cholesky decomposition

# Parameters
beta_BC = 7.94
sigma_bc = 0.78  # 0.68
delta_layer = 0.1  # 5*dx; %6*dx;

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

RHS = np.zeros_like(Xint)

Test_Phi = spsolve(Lap, RHS.flatten(order='F') - Phi_BCs.flatten(order='F'))
Test_Npm = spsolve(Lap, RHS.flatten(order='F') - Npm_BCs.flatten(order='F'))

fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(Xint, Yint, Test_Phi.reshape(Ny, Nx), cmap='turbo')
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(Xint, Yint, Test_Npm.reshape(Ny, Nx), cmap='turbo')
plt.show()

# Make immersed boundary mats
rad = 0.25
dth = dx / rad
theta = np.arange(0, 2*np.pi - dth, dth)
Nib = len(theta)
xib = rad * np.cos(theta) 
yib = rad * np.sin(theta)
n_x = np.cos(theta)
n_y = np.sin(theta)

cut = 6 * 1.2 * dx # cutoff value

# Compute exact solutions
Phi_BC = Phi_exact(X, Y)
N_pm_BC = Npm_exact(X, Y)

# Boundary conditions context
ctxt_BCs = np.concatenate([
    Phi_BCs.flatten(order='F'),
    Npm_BCs.flatten(order='F'),
    Npm_BCs.flatten(order='F'),
    np.zeros(len(xib)) - (sigma_bc/delta_layer),
    np.zeros(len(xib)),
    np.zeros(len(xib))
])

# Delta functions
@jit(nopython=True)
def delta_a(r, a):
    return (1/(2*np.pi*a**2)) * np.exp(-0.5*(r/a)**2)

@jit(nopython=True)
def delta(r):
    return delta_a(r, 1.2*dx)

@jit(nopython=True)
def delta_r(r):
    return (1/(1.2*dx))**2 * r * delta_a(r, 1.2*dx)

# Spreading operator
@jit(nopython=True)
def spreadQ_prime(X, Y, xq, yq, n_x, n_y, q, delta_r, cut):
    Sq = np.zeros_like(X)
    Nq = len(q)

    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    Nx = X.shape[1]
    Ny = Y.shape[0]

    for k in range(Nq):
        xk = xq[k]
        yk = yq[k]
        
        i_min = max(int((xk - cut - X[0, 0]) / dx), 0)
        i_max = min(int((xk + cut - X[0, 0]) / dx) + 1, Nx)
        j_min = max(int((yk - cut - Y[0, 0]) / dy), 0)
        j_max = min(int((yk + cut - Y[0, 0]) / dy) + 1, Ny)
        
        X_local = X[j_min:j_max, i_min:i_max]
        Y_local = Y[j_min:j_max, i_min:i_max]
        
        Rk = np.sqrt((X_local - xk)**2 + (Y_local - yk)**2)
        
        mask = (Rk <= cut)
        
        n_dot_rhat = np.where(mask,
            (n_x[k] * (X_local - xk) + n_y[k] * (Y_local - yk)) / Rk,
            0.0
        )
        
        contribution = q[k] * n_dot_rhat * delta_r(Rk) * mask
        Sq[j_min:j_max, i_min:i_max] += contribution
    
    return Sq

# Interpolation operator
@jit(nopython=True)
def interpPhi_prime(X, Y, xq, yq, n_x, n_y, Phi, delta_r, cut):
    Jphi = np.zeros_like(xq)
    dx_loc = X[0, 1] - X[0, 0]
    dy_loc = Y[1, 0] - Y[0, 0]
    Ny, Nx = X.shape

    for k in range(len(xq)):
        xk, yk = xq[k], yq[k]
        nxk, nyk = n_x[k], n_y[k]

        # Find the grid region affected by this point
        i_min = max(int((xk - cut - X[0,0]) / dx_loc), 0)
        i_max = min(int((xk + cut - X[0,0]) / dx_loc) + 1, Nx)
        j_min = max(int((yk - cut - Y[0,0]) / dy_loc), 0)
        j_max = min(int((yk + cut - Y[0,0]) / dy_loc) + 1, Ny)

        # Extract local patch
        X_local = X[j_min:j_max, i_min:i_max]
        Y_local = Y[j_min:j_max, i_min:i_max]
        Phi_local = Phi[j_min:j_max, i_min:i_max]

        dx = X_local - xk
        dy = Y_local - yk
        R = np.sqrt(dx**2 + dy**2)
        mask = R <= cut

        n_dot_rhat = np.where(mask, (nxk * dx + nyk * dy) / R, 0.0)

        delta_vals = delta_r(R)
        contribution = Phi_local * n_dot_rhat * delta_vals * mask

        Jphi[k] = dx_loc * dy_loc * np.sum(contribution)

    return Jphi

# @jit(nopython=True)
# def spreadQ(X, Y, xq, yq, q, delta, cut):
#     Sq = np.zeros_like(X)
#     Nq = len(q)
    
#     dx = X[0, 1] - X[0, 0]
#     dy = Y[1, 0] - Y[0, 0]
#     Nx = X.shape[1]
#     Ny = Y.shape[0]

#     for k in range(Nq):
#         xk = xq[k]
#         yk = yq[k]
        
#         # Compute local window indices
#         i_min = max(int((xk - cut - X[0, 0]) / dx), 0)
#         i_max = min(int((xk + cut - X[0, 0]) / dx) + 1, Nx)
#         j_min = max(int((yk - cut - Y[0, 0]) / dy), 0)
#         j_max = min(int((yk + cut - Y[0, 0]) / dy) + 1, Ny)
        
#         # Extract local window
#         X_local = X[j_min:j_max, i_min:i_max]
#         Y_local = Y[j_min:j_max, i_min:i_max]
        
#         Rk = np.sqrt((X_local - xk)**2 + (Y_local - yk)**2)
#         mask = (Rk <= cut)
        
#         contribution = q[k] * delta(Rk) * mask
        
#         Sq[j_min:j_max, i_min:i_max] += contribution
    
#     return Sq

@jit(nopython=True)
def interpPhi(X, Y, xq, yq, Phi, delta, cut):
    Jphi = np.zeros_like(xq)
    dx_loc = X[0, 1] - X[0, 0]
    dy_loc = Y[1, 0] - Y[0, 0]
    
    for k in range(len(xq)):
        xk, yk = xq[k], yq[k]
        nxk, nyk = n_x[k], n_y[k]

        # Find the grid region affected by this point
        i_min = max(int((xk - cut - X[0,0]) / dx_loc), 0)
        i_max = min(int((xk + cut - X[0,0]) / dx_loc) + 1, Nx)
        j_min = max(int((yk - cut - Y[0,0]) / dy_loc), 0)
        j_max = min(int((yk + cut - Y[0,0]) / dy_loc) + 1, Ny)

        # Extract local patch
        X_local = X[j_min:j_max, i_min:i_max]
        Y_local = Y[j_min:j_max, i_min:i_max]
        Phi_local = Phi[j_min:j_max, i_min:i_max]

        dx = X_local - xk
        dy = Y_local - yk
        R = np.sqrt(dx**2 + dy**2)
        mask = R <= cut

        delta_vals = delta(R)
        contribution = Phi_local * delta_vals * mask

        Jphi[k] = dx_loc * dy_loc * np.sum(contribution)

    return Jphi

def Sop_prime(q):
    return spreadQ_prime(Xint, Yint, xib, yib, n_x, n_y, q, delta_r, cut)

def Jop_prime(P):
    return interpPhi_prime(Xint, Yint, xib, yib, n_x, n_y, P, delta_r, cut)

def Jop(P):
    return interpPhi(Xint, Yint, xib, yib, P, delta, cut)

def G_d_G(Phi, N_pm):
    return Grad_dot_Grad(Phi, N_pm, dx, dy, Nx, Ny, Phi_BC, N_pm_BC)

def AxOp_prev(ctxt, ctxt_prev):
    return Constrained_Lap(ctxt, ctxt_prev, dLap, delta_layer, Nx, Ny, Nib, Sop_prime, Jop_prime)

def b_Op(ctxt):
    return Build_RHS(ctxt, ctxt_BCs, Lap, dLap, G_d_G, delta_layer, Nx, Ny, Nib, Jop, Jop_prime)

def schurOp_prev(p_blocks):
    return apply_Schur(dLap, p_blocks, delta_layer)
#@jit(nopython=True)
def Grad_dot_Grad(Phi, N_pm, dx, dy, Nx, Ny, Phi_BC, N_pm_BC):
    Phi = Phi.reshape(Ny, Nx).T
    N_pm = N_pm.reshape(Ny, Nx).T

    Phi_BC_y = np.vstack([Phi_BC[0, 1:-1].reshape(1, -1), Phi, Phi_BC[-1, 1:-1].reshape(1, -1)])
    Phi_y = (0.5/dy) * (Phi_BC_y[2:, :] - Phi_BC_y[:-2, :])

    N_pm_BC_y = np.vstack([N_pm_BC[0, 1:-1].reshape(1, -1), N_pm, N_pm_BC[-1, 1:-1].reshape(1, -1)])
    N_pm_y = (0.5/dy) * (N_pm_BC_y[2:, :] - N_pm_BC_y[:-2, :])

    Phi_BC_x = np.hstack([Phi_BC[1:-1, 0].reshape(-1, 1), Phi, Phi_BC[1:-1, -1].reshape(-1, 1)])
    Phi_x = (0.5/dx) * (Phi_BC_x[:, 2:] - Phi_BC_x[:, :-2])

    N_pm_BC_x = np.hstack([N_pm_BC[1:-1, 0].reshape(-1, 1), N_pm, N_pm_BC[1:-1, -1].reshape(-1, 1)])
    N_pm_x = (0.5/dx) * (N_pm_BC_x[:, 2:] - N_pm_BC_x[:, :-2])

    G_d_G = N_pm_x * Phi_x + N_pm_y * Phi_y
    return G_d_G.flatten(order='F')

def Constrained_Lap(ctxt, ctxt_prev, dLap, delta_layer, Nx, Ny, Nib, Sop_prime, Jop_prime):
    A_x_Ctx = np.zeros_like(ctxt)
    
    sz = Nx * Ny
    Phi = ctxt[:sz]
    N_p = ctxt[sz:2*sz]
    N_m = ctxt[2*sz:3*sz]
    q_i = 3 * sz
    Q = ctxt[q_i:q_i+Nib]
    Q_p = ctxt[q_i+Nib:q_i+2*Nib]
    Q_m = ctxt[q_i+2*Nib:q_i+3*Nib]
    
    SQ = Sop_prime(Q)
    SQ_p = Sop_prime(Q_p)
    SQ_m = Sop_prime(Q_m)
    
    dl2 = delta_layer**2

    A_x_Ctx[:sz] = dl2 * Phi - dLap.solve_A(0.5*N_p - 0.5*N_m + SQ.flatten(order='F'))
    A_x_Ctx[sz:2*sz] = N_p - dLap.solve_A(SQ_p.flatten(order='F'))
    A_x_Ctx[2*sz:3*sz] = N_m - dLap.solve_A(SQ_m.flatten(order='F'))
    A_x_Ctx[q_i:q_i+Nib] = Jop_prime(Phi.reshape(Ny, Nx, order='F'))
    A_x_Ctx[q_i+Nib:q_i+2*Nib] = Jop_prime(N_p.reshape(Ny, Nx, order='F'))
    A_x_Ctx[q_i+2*Nib:q_i+3*Nib] = Jop_prime(N_m.reshape(Ny, Nx, order='F'))
    
    return A_x_Ctx

def apply_Schur(dLap, p_blocks, delta_layer):
    p, p_p, p_m = p_blocks

    # apply B 
    Bp = Sop_prime(p)
    Bp_p = Sop_prime(p_p)
    Bp_m = Sop_prime(p_m)

    # apply A inverse 
    Ainv_Bp, Ainv_Bp_p, Ainv_Bp_m = apply_Ainv(dLap, [Bp, Bp_p, Bp_m], delta_layer)

    # apply C 
    res_1 = delta_layer * Jop_prime(Ainv_Bp.reshape(Ny, Nx, order='F'))
    res_2 = Jop_prime(Ainv_Bp_p.reshape(Ny, Nx, order='F'))
    res_3 = Jop_prime(Ainv_Bp_m.reshape(Ny, Nx, order='F'))

    return [res_1, res_2, res_3]

def apply_Ainv(dLap, target_vec, delta_layer):
    target_vec_1, target_vec_2, target_vec_3 = target_vec

    dl2 = delta_layer**2

    # second and third blocks are straightforward
    result_vec_2 = -dLap.solve_A(target_vec_2.flatten(order='F'))
    result_vec_3 = -dLap.solve_A(target_vec_3.flatten(order='F'))

    # use these results to compute first block 
    rhs = target_vec_1.flatten(order='F') + 0.5 * result_vec_2 - 0.5 * result_vec_3
    result_vec_1 = -dLap.solve_A(rhs / dl2)

    return [result_vec_1, result_vec_2, result_vec_3]

def Build_RHS(ctxt, ctxt_BCs, Lap, dLap, G_d_G, delta_layer, Nx, Ny, Nib, Jop, Jop_prime):
    b_Ctx = np.zeros_like(ctxt_BCs)
    
    sz = Nx * Ny
    q_i = 3 * sz
    Phi = ctxt[:sz]
    N_p = ctxt[sz:2*sz]
    N_m = ctxt[2*sz:3*sz]
    
    Phi_BC = ctxt_BCs[:sz]
    N_p_BC = ctxt_BCs[sz:2*sz]
    N_m_BC = ctxt_BCs[2*sz:3*sz]
    Q_BC = ctxt_BCs[q_i:q_i+Nib]
    Q_p_BC = ctxt_BCs[q_i+Nib:q_i+2*Nib]
    Q_m_BC = ctxt_BCs[q_i+2*Nib:q_i+3*Nib]
    
    dl2 = delta_layer**2
    
    computed_lap = (-Lap) @ Phi
    computed_lap = computed_lap + Phi_BC

    b_Ctx[:sz] =  -dLap.solve_A(-dl2 * Phi_BC)
    b_Ctx[sz:2*sz] =  -dLap.solve_A(-N_p * computed_lap - N_p_BC - G_d_G(Phi, N_p))
    b_Ctx[2*sz:3*sz] =  -dLap.solve_A(N_m * computed_lap - N_m_BC + G_d_G(Phi, N_m))
    b_Ctx[q_i:q_i+Nib] = Q_BC
    b_Ctx[q_i+Nib:q_i+2*Nib] = Q_p_BC - Jop(N_p.reshape(Ny, Nx, order='F')) * Jop_prime(Phi.reshape(Ny, Nx, order='F'))
    b_Ctx[q_i+2*Nib:q_i+3*Nib] = Q_m_BC + Jop(N_m.reshape(Ny, Nx, order='F')) * Jop_prime(Phi.reshape(Ny, Nx, order='F'))
    
    return b_Ctx

class ConstrainedLapOperator:
    def __init__(self, dLap, delta_layer, Nx, Ny, Nib, Sop_prime, Jop_prime):
        self.dLap = dLap
        self.delta_layer = delta_layer
        self.Nx = Nx
        self.Ny = Ny
        self.Nib = Nib
        self.Sop_prime = Sop_prime
        self.Jop_prime = Jop_prime
        self.ctxt_prev = None
    
    def set_context(self, ctxt_prev):
        self.ctxt_prev = ctxt_prev.copy()  # Make a copy to avoid reference issues
    
    def matvec(self, xx):
        return Constrained_Lap(xx, self.ctxt_prev, self.dLap, self.delta_layer, 
                              self.Nx, self.Ny, self.Nib, self.Sop_prime, self.Jop_prime)
    
# LinearOperator object for using the Schur complement as our LHS matrix in GMRES
def SchurLinearOperator(dLap, shape, Nib, delta_layer):
    n = shape
    def mv(p_block):
        p = p_block[0:Nib]
        p_p = p_block[Nib:2*Nib]
        p_m = p_block[2*Nib:3*Nib]

        res_1, res_2, res_3 = apply_Schur(dLap, [p, p_p, p_m], delta_layer)
        return np.concatenate([res_1, res_2, res_3])
    return LinearOperator((n, n), matvec=mv)

def schur_rhs(dLap, rhs, Nx, Ny, Nib, delta_layer):
    sz = Nx * Ny
    q_i = 3 * sz

    rhs_1 = rhs[:sz]
    rhs_2 = rhs[sz:2*sz]
    rhs_3 = rhs[2*sz:3*sz]
    rhs_4 = rhs[q_i:q_i+Nib]
    rhs_5 = rhs[q_i+Nib:q_i+2*Nib]
    rhs_6 = rhs[q_i+2*Nib:q_i+3*Nib]

    Ainv_phi, Ainv_n_p, Ainv_n_m = apply_Ainv(dLap, [rhs_1, rhs_2, rhs_3], delta_layer)
    CAinv_phi = delta_layer * Jop_prime(Ainv_phi.reshape(Ny, Nx, order='F'))
    CAinv_n_p = Jop_prime(Ainv_n_p.reshape(Ny, Nx, order='F'))
    CAinv_n_m = Jop_prime(Ainv_n_m.reshape(Ny, Nx, order='F'))

    schur_rhs_1 = CAinv_phi - rhs_4
    schur_rhs_2 = CAinv_n_p - rhs_5
    schur_rhs_3 = CAinv_n_m - rhs_6

    return np.concatenate((schur_rhs_1, schur_rhs_2, schur_rhs_3))

def post_processing_compute(dLap, p_block, rhs, Nx, Ny, Nib, delta_layer):
    sz = Nx * Ny

    p = p_block[0:Nib]
    p_p = p_block[Nib:2*Nib]
    p_m = p_block[2*Nib:3*Nib]

    rhs_1 = rhs[:sz]
    rhs_2 = rhs[sz:2*sz]
    rhs_3 = rhs[2*sz:3*sz]

    dl2 = delta_layer**2

    # get rhs for what we already know how to solve
    rhs_n_p = rhs_2 - Sop_prime(p_p).flatten(order='F')
    rhs_n_m = rhs_3 - Sop_prime(p_m).flatten(order='F')

    n_p = -dLap.solve_A(rhs_n_p)
    n_m = -dLap.solve_A(rhs_n_m)

    rhs_phi = (rhs_1 + 0.5*n_p - 0.5*n_m - Sop_prime(p).flatten(order='F')) / dl2
    phi = -dLap.solve_A(rhs_phi)

    return np.concatenate((phi, n_p, n_m, p, p_p, p_m))

# Create the operator once
lap_operator = ConstrainedLapOperator(dLap, delta_layer, Nx, Ny, Nib, Sop_prime, Jop_prime)

# Load initial conditions from .mat file
ld = loadmat('BC_run_N_300_r0p25.mat')
METHOD = 'cubic'  # equivalent to 'makima' in MATLAB

Ny_ld = int(ld['Ny'][0, 0])
Nx_ld = int(ld['Nx'][0, 0])
Nib_ld = int(ld['Nib'][0, 0])
sz = Ny_ld * Nx_ld

ctxt_ld = ld['ctxt'].flatten(order='F')
Phi_ld = ctxt_ld[:sz].reshape(Ny_ld, Nx_ld, order='F')  # Important: use Fortran order
N_p_ld = ctxt_ld[sz:2*sz].reshape(Ny_ld, Nx_ld, order='F')
N_m_ld = ctxt_ld[2*sz:3*sz].reshape(Ny_ld, Nx_ld, order='F')
Q_ld = ctxt_ld[3*sz:3*sz+Nib_ld]
Q_p_ld = ctxt_ld[3*sz+Nib_ld:3*sz+2*Nib_ld]
Q_m_ld = ctxt_ld[3*sz+2*Nib_ld:3*sz+3*Nib_ld]

Xint_ld = ld['Xint']
Yint_ld = ld['Yint']
theta_ld = ld['theta'].flatten()

# Extract the coordinate vectors from the loaded grid
x_ld = Xint_ld[0, :]  # First row gives x-coordinates
y_ld = Yint_ld[:, 0]  # First column gives y-coordinates

ldData = loadmat('data.mat')
Phi_init = ldData['Phi_init']
N_p_init = ldData['N_p_init']
N_m_init = ldData['N_m_init']

#Phi_init = interpn((x_ld, y_ld), Phi_ld, (Xint.T, Yint.T), method='linear', bounds_error=False, fill_value=None)

#N_p_init = interpn((x_ld, y_ld), N_p_ld, (Xint.T, Yint.T), method='nearest', bounds_error=False, fill_value=None)
#N_m_init = interpn((x_ld, y_ld), N_m_ld, (Xint.T, Yint.T), method='nearest', bounds_error=False, fill_value=None)

#N_p_init_f = RegularGridInterpolator((x_ld, y_ld), N_p_ld, 
#                                    method=METHOD, bounds_error=False, fill_value=None)
#N_m_init_f = RegularGridInterpolator((x_ld, y_ld), N_m_ld, 
#                                    method=METHOD, bounds_error=False, fill_value=None)

#points_new = np.column_stack([Xint.flatten(order='F'), Yint.flatten(order='F')])

#N_p_init = N_p_init_f(points_new).reshape(Ny, Nx)
#N_m_init = N_m_init_f(points_new).reshape(Ny, Nx)

# Interpolate boundary quantities
Q_init = Akima1DInterpolator(theta_ld, Q_ld, method="makima", extrapolate=True)(theta)
Q_p_init = Akima1DInterpolator(theta_ld, Q_p_ld, method="makima", extrapolate=True)(theta)
Q_m_init = Akima1DInterpolator(theta_ld, Q_m_ld, method="makima", extrapolate=True)(theta)

ctxt = np.concatenate([
    Phi_init.flatten(order='F'),
    N_p_init.flatten(order='F'),
    N_m_init.flatten(order='F'),
    Q_init,
    Q_p_init,
    Q_m_init
])

# Check initial residual
RHS = b_Op(ctxt)
err_init = np.linalg.norm(AxOp_prev(ctxt, ctxt) - RHS) / np.linalg.norm(RHS)
print(f'Initial residual: {err_init}')

# Anderson acceleration parameters
beta = 0.2
m = 50
DU = np.full((len(RHS), m), np.nan)
DG = np.full((len(RHS), m), np.nan)

tol = 1e-5
u_n = ctxt.copy()

RHS_schur = schur_rhs(dLap, RHS, Nx, Ny, Nib, delta_layer)
p_guess = u_n[3*Nx*Ny:]

schurOp = SchurLinearOperator(dLap, Nib*3, Nib, delta_layer)

# Initial GMRES solve
p_next, info = gmres(schurOp, RHS_schur, rtol=tol, maxiter=1000, restart=500, x0=p_guess, callback=lambda x: print(f"GMRES residual: {np.linalg.norm(x)}"))
if info != 0:
    print(f'GMRES warning: convergence info = {info}')

G_u_n = post_processing_compute(dLap, p_next, RHS, Nx, Ny, Nib, delta_layer)
u_next = G_u_n.copy()
G_u_next = G_u_n.copy()
err = []

# Anderson acceleration loop
for its in range(100000):
    RHS = b_Op(u_next)
    RHS_schur = schur_rhs(dLap, RHS, Nx, Ny, Nib, delta_layer)
    lap_operator.set_context(u_next)  # Update the context
    schurOp = SchurLinearOperator(dLap, Nib*3, Nib, delta_layer)

    p_next, info = gmres(schurOp, RHS_schur, rtol=tol, maxiter=1000, restart=500, x0=p_guess, callback=lambda x: print(f"GMRES residual: {np.linalg.norm(x)}"))
    if info != 0:
        print(f'GMRES warning: convergence info = {info}')

    G_u_next = post_processing_compute(dLap, p_next, RHS, Nx, Ny, Nib, delta_layer)

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
    Q_qr, R_qr = qr(DF, mode='economic')
    gamma = np.linalg.solve(R_qr, Q_qr.T @ f_n)
    
    u_n = u_next.copy()
    G_u_n = G_u_next.copy()
    
    u_next = (G_u_next - DG[:, :m_n] @ gamma) - (1-beta) * (f_n - DF @ gamma)
    
    # Extract solution components
    Phi = u_next[:Ny*Nx].reshape(Ny, Nx, order='F')
    Np = u_next[Ny*Nx:2*Nx*Ny].reshape(Ny, Nx, order='F')
    Nm = u_next[2*Ny*Nx:3*Nx*Ny].reshape(Ny, Nx, order='F')
    
    p_next = u_next[3*Nx*Ny:3*Nx*Ny+Nib]
    p_p_next = u_next[3*Nx*Ny+Nib:3*Nx*Ny+2*Nib]
    p_m_next = u_next[3*Nx*Ny+2*Nib:]

    err_res_1, err_res_2, err_res_3 = apply_Schur(
        dLap, [p_next, p_p_next, p_m_next], delta_layer
    )

    # update guess and RHS 
    RHS = b_Op(u_next)
    p_guess = u_next[3*Nx*Ny:]

    # Check convergence
    err_curr = np.linalg.norm(AxOp_prev(u_next, u_next) - RHS) / np.linalg.norm(RHS)
    err.append(err_curr)
    
    print(f'Iteration {its}: residual = {err_curr}')
    
    if err_curr < 1e-4:
        print('Converged!')
        break
    
    # Plot current solution
    # if its % 10 == 0:  # Plot every 10 iterations
    #     plt.clf()
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     surf = ax.plot_surface(Xint, Yint, Np, cmap='turbo', alpha=0.8)
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_title(r'$N_+$')
    #     plt.pause(0.01)

    #break

ctxt_final = u_next.copy()

# Save results
savemat('Err_Run_N_450.mat', {
    'ctxt': ctxt_final,
    'u_next': u_next,
    'Xint': Xint,
    'Yint': Yint,
    'xib': xib,
    'yib': yib,
    'Phi': Phi,
    'Np': Np,
    'Nm': Nm,
    'err': np.array(err)
})

print("Computation completed and results saved!")

elapsed = time.time() - t

print(f'Elapsed time: {elapsed}')