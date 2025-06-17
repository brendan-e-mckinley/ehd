import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, eye, kron
from scipy.sparse.linalg import spsolve, gmres
from scipy.linalg import qr
from scipy.io import loadmat, savemat
from scipy.interpolate import griddata, interp1d

# Set up plotting parameters (equivalent to MATLAB set commands)
plt.rcParams.update({
    'font.size': 35,
    'lines.linewidth': 3,
    'axes.labelsize': 35,
    'xtick.labelsize': 35,
    'ytick.labelsize': 35
})

# Setup grid in x-y
Nx = 450  # 256; % number of grid point along one direction
plt.clf()
L = 2.0 * np.pi  # 2.0*pi
x = np.linspace(-L/2, L/2, Nx+2)  # periodic grid
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
Lap = kron(I_nx, D2_d) + kron(D2_d, I_ny)

# Parameters
beta_BC = 7.94
sigma_bc = 0.78  # 0.68
delta_layer = 0.1  # 5*dx; %6*dx;

# Exact solutions
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

Test_Phi = spsolve(Lap, RHS.flatten() - Phi_BCs.flatten())
Test_Npm = spsolve(Lap, RHS.flatten() - Npm_BCs.flatten())

fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(Xint, Yint, Test_Phi.reshape(Ny, Nx), cmap='turbo')
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(Xint, Yint, Test_Npm.reshape(Ny, Nx), cmap='turbo')
plt.show()

# Make immersed boundary mats
rad = 0.25
dth = dx / rad
theta = np.arange(0, 2*np.pi, dth)
Nib = len(theta)
xib = rad * np.cos(theta)
yib = rad * np.sin(theta)
n_x = np.cos(theta)
n_y = np.sin(theta)

# Delta functions
def delta_a(r, a):
    return (1/(2*np.pi*a**2)) * np.exp(-0.5*(r/a)**2)

def delta(r):
    return delta_a(r, 1.2*dx)

def delta_r(r):
    return (1/(1.2*dx))**2 * r * delta_a(r, 1.2*dx)

cut = 6 * 1.2 * dx

# Define operators
def spreadQ_prime(X, Y, xq, yq, n_x, n_y, q, delta_r, cut):
    Sq = np.zeros_like(X)
    Nq = len(q)
    for k in range(Nq):
        Rk = np.sqrt((X - xq[k])**2 + (Y - yq[k])**2)
        mask = (Rk <= cut)
        n_dot_rhat = (n_x[k]*(X[mask] - xq[k]) + n_y[k]*(Y[mask] - yq[k])) / Rk[mask]
        Sq[mask] += q[k] * n_dot_rhat * delta_r(Rk[mask])
    return Sq

def interpPhi_prime(X, Y, xq, yq, n_x, n_y, Phi, delta_r, cut):
    Jphi = np.zeros_like(xq)
    Nq = len(xq)
    dx_loc = X[0, 1] - X[0, 0]
    dy_loc = Y[1, 0] - Y[0, 0]
    for k in range(Nq):
        Rk = np.sqrt((X - xq[k])**2 + (Y - yq[k])**2)
        mask = (Rk <= cut)
        n_dot_rhat = (n_x[k]*(X[mask] - xq[k]) + n_y[k]*(Y[mask] - yq[k])) / Rk[mask]
        Jphi[k] = dx_loc * dy_loc * np.sum(Phi[mask] * n_dot_rhat * delta_r(Rk[mask]))
    return Jphi

def spreadQ(X, Y, xq, yq, q, delta, cut):
    Sq = np.zeros_like(X)
    Nq = len(q)
    for k in range(Nq):
        Rk = np.sqrt((X - xq[k])**2 + (Y - yq[k])**2)
        mask = (Rk <= cut)
        Sq[mask] += q[k] * delta(Rk[mask])
    return Sq

def interpPhi(X, Y, xq, yq, Phi, delta, cut):
    Jphi = np.zeros_like(xq)
    Nq = len(xq)
    dx_loc = X[0, 1] - X[0, 0]
    dy_loc = Y[1, 0] - Y[0, 0]
    for k in range(Nq):
        Rk = np.sqrt((X - xq[k])**2 + (Y - yq[k])**2)
        mask = (Rk <= cut)
        Jphi[k] = dx_loc * dy_loc * np.sum(Phi[mask] * delta(Rk[mask]))
    return Jphi

# Define lambda functions for operators
Sop_prime = lambda q: spreadQ_prime(Xint, Yint, xib, yib, n_x, n_y, q, delta_r, cut)
Jop_prime = lambda P: interpPhi_prime(Xint, Yint, xib, yib, n_x, n_y, P, delta_r, cut)
Sop = lambda q: spreadQ(Xint, Yint, xib, yib, q, delta, cut)
Jop = lambda P: interpPhi(Xint, Yint, xib, yib, P, delta, cut)

Phi_BC = Phi_exact(X, Y)
N_pm_BC = Npm_exact(X, Y)

def Grad_dot_Grad(Phi, N_pm, dx, dy, Nx, Ny, Phi_BC, N_pm_BC):
    # Phi_x * N_x + Phi_y * N_y
    Phi = Phi.reshape(Ny, Nx)
    N_pm = N_pm.reshape(Ny, Nx)
    
    Phi_BC_y = np.vstack([Phi_BC[0, 1:-1].reshape(1, -1), Phi, Phi_BC[-1, 1:-1].reshape(1, -1)])
    Phi_y = (0.5/dy) * (Phi_BC_y[2:, :] - Phi_BC_y[:-2, :])
    
    N_pm_BC_y = np.vstack([N_pm_BC[0, 1:-1].reshape(1, -1), N_pm, N_pm_BC[-1, 1:-1].reshape(1, -1)])
    N_pm_y = (0.5/dy) * (N_pm_BC_y[2:, :] - N_pm_BC_y[:-2, :])
    
    Phi_BC_x = np.hstack([Phi_BC[1:-1, 0].reshape(-1, 1), Phi, Phi_BC[1:-1, -1].reshape(-1, 1)])
    Phi_x = (0.5/dx) * (Phi_BC_x[:, 2:] - Phi_BC_x[:, :-2])
    
    N_pm_BC_x = np.hstack([N_pm_BC[1:-1, 0].reshape(-1, 1), N_pm, N_pm_BC[1:-1, -1].reshape(-1, 1)])
    N_pm_x = (0.5/dx) * (N_pm_BC_x[:, 2:] - N_pm_BC_x[:, :-2])
    
    G_d_G = N_pm_x * Phi_x + N_pm_y * Phi_y
    return G_d_G.flatten()

G_d_G = lambda Phi, N_pm: Grad_dot_Grad(Phi, N_pm, dx, dy, Nx, Ny, Phi_BC, N_pm_BC)

# Boundary conditions context
ctxt_BCs = np.concatenate([
    Phi_BCs.flatten(),
    Npm_BCs.flatten(),
    Npm_BCs.flatten(),
    np.zeros(len(xib)) - (sigma_bc/delta_layer),
    np.zeros(len(xib)),
    np.zeros(len(xib))
])

def Constrained_Lap(ctxt, ctxt_prev, Lap, delta_layer, Nx, Ny, Nib, Sop, Jop, Sop_prime, Jop_prime):
    A_x_Ctx = np.zeros_like(ctxt)
    
    sz = Nx * Ny
    Phi = ctxt[:sz]
    N_p = ctxt[sz:2*sz]
    N_m = ctxt[2*sz:3*sz]
    q_i = 3 * sz
    Q = ctxt[q_i:q_i+Nib]
    Q_p = ctxt[q_i+Nib:q_i+2*Nib]
    Q_m = ctxt[q_i+2*Nib:q_i+3*Nib]
    
    Phi_prev = ctxt_prev[:sz]
    N_p_prev = ctxt_prev[sz:2*sz]
    N_m_prev = ctxt_prev[2*sz:3*sz]
    
    SQ = Sop_prime(Q)
    SQ_p = Sop_prime(Q_p)
    SQ_m = Sop_prime(Q_m)
    
    dl2 = delta_layer**2
    A_x_Ctx[:sz] = dl2 * Phi + spsolve(Lap, 0.5*N_p - 0.5*N_m + SQ.flatten())
    A_x_Ctx[sz:2*sz] = N_p + spsolve(Lap, SQ_p.flatten())
    A_x_Ctx[2*sz:3*sz] = N_m + spsolve(Lap, SQ_m.flatten())
    A_x_Ctx[q_i:q_i+Nib] = Jop_prime(Phi.reshape(Ny, Nx))
    A_x_Ctx[q_i+Nib:q_i+2*Nib] = Jop_prime(N_p.reshape(Ny, Nx))
    A_x_Ctx[q_i+2*Nib:q_i+3*Nib] = Jop_prime(N_m.reshape(Ny, Nx))
    
    return A_x_Ctx

def Build_RHS(ctxt, ctxt_BCs, Lap, G_d_G, delta_layer, dx, dy, Nx, Ny, Nib, Sop, Jop, Sop_prime, Jop_prime):
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
    b_Ctx[:sz] = spsolve(Lap, -dl2 * Phi_BC)
    
    b_Ctx[sz:2*sz] = spsolve(Lap, -N_p * (Lap @ Phi + Phi_BC) - N_p_BC - G_d_G(Phi, N_p))
    b_Ctx[2*sz:3*sz] = spsolve(Lap, N_m * (Lap @ Phi + Phi_BC) - N_m_BC + G_d_G(Phi, N_m))
    b_Ctx[q_i:q_i+Nib] = Q_BC
    b_Ctx[q_i+Nib:q_i+2*Nib] = Q_p_BC - Jop(N_p.reshape(Ny, Nx)) * Jop_prime(Phi.reshape(Ny, Nx))
    b_Ctx[q_i+2*Nib:q_i+3*Nib] = Q_m_BC + Jop(N_m.reshape(Ny, Nx)) * Jop_prime(Phi.reshape(Ny, Nx))
    
    return b_Ctx

# Load initial conditions from .mat file
ld = loadmat('BC_run_N_300_r0p25.mat')
METHOD = 'cubic'  # equivalent to 'makima' in MATLAB
EMETHOD = 'nearest'

Ny_ld = int(ld['Ny'][0, 0])
Nx_ld = int(ld['Nx'][0, 0])
Nib_ld = int(ld['Nib'][0, 0])
sz = Ny_ld * Nx_ld

ctxt_ld = ld['ctxt'].flatten()
Phi_ld = ctxt_ld[:sz].reshape(Ny_ld, Nx_ld)
N_p_ld = ctxt_ld[sz:2*sz].reshape(Ny_ld, Nx_ld)
N_m_ld = ctxt_ld[2*sz:3*sz].reshape(Ny_ld, Nx_ld)
Q_ld = ctxt_ld[3*sz:3*sz+Nib_ld]
Q_p_ld = ctxt_ld[3*sz+Nib_ld:3*sz+2*Nib_ld]
Q_m_ld = ctxt_ld[3*sz+2*Nib_ld:3*sz+3*Nib_ld]

Xint_ld = ld['Xint']
Yint_ld = ld['Yint']
theta_ld = ld['theta'].flatten()

# Interpolate to current grid
points_ld = np.column_stack([Xint_ld.flatten(), Yint_ld.flatten()])
points_new = np.column_stack([Xint.flatten(), Yint.flatten()])

Phi_init = griddata(points_ld, Phi_ld.flatten(), points_new, method=METHOD, fill_value=0).reshape(Ny, Nx)
N_p_init = griddata(points_ld, N_p_ld.flatten(), points_new, method=METHOD, fill_value=1).reshape(Ny, Nx)
N_m_init = griddata(points_ld, N_m_ld.flatten(), points_new, method=METHOD, fill_value=1).reshape(Ny, Nx)

# Interpolate boundary quantities
Q_init = interp1d(theta_ld, Q_ld, kind=METHOD, bounds_error=False, fill_value='extrapolate')(theta)
Q_p_init = interp1d(theta_ld, Q_p_ld, kind=METHOD, bounds_error=False, fill_value='extrapolate')(theta)
Q_m_init = interp1d(theta_ld, Q_m_ld, kind=METHOD, bounds_error=False, fill_value='extrapolate')(theta)

ctxt = np.concatenate([
    Phi_init.flatten(),
    N_p_init.flatten(),
    N_m_init.flatten(),
    Q_init,
    Q_p_init,
    Q_m_init
])

# Define operators for iteration
AxOp_prev = lambda ctxt, ctxt_prev: Constrained_Lap(ctxt, ctxt_prev, Lap, delta_layer, Nx, Ny, Nib, Sop, Jop, Sop_prime, Jop_prime)
b_Op = lambda ctxt: Build_RHS(ctxt, ctxt_BCs, Lap, G_d_G, delta_layer, dx, dy, Nx, Ny, Nib, Sop, Jop, Sop_prime, Jop_prime)

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
RHS = b_Op(ctxt)
AxOp = lambda xx: AxOp_prev(xx, ctxt)

# Initial GMRES solve
G_u_n, info = gmres(AxOp, RHS, rtol=tol, maxiter=1000, x0=u_n)
if info != 0:
    print(f'GMRES warning: convergence info = {info}')

u_next = G_u_n.copy()
G_u_next = G_u_n.copy()
err = []

# Anderson acceleration loop
for its in range(100000):
    RHS = b_Op(u_next)
    AxOp = lambda xx: AxOp_prev(xx, u_next)
    G_u_next, info = gmres(AxOp, RHS, tol=tol, maxiter=1000, x0=u_next)
    
    if info != 0:
        print(f'GMRES warning at iteration {its}: convergence info = {info}')
    
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
    Phi = u_next[:Ny*Nx].reshape(Ny, Nx)
    Np = u_next[Ny*Nx:2*Nx*Ny].reshape(Ny, Nx)
    Nm = u_next[2*Ny*Nx:3*Nx*Ny].reshape(Ny, Nx)
    
    # Check convergence
    RHS = b_Op(u_next)
    err_curr = np.linalg.norm(AxOp_prev(u_next, u_next) - RHS) / np.linalg.norm(RHS)
    err.append(err_curr)
    
    print(f'Iteration {its}: residual = {err_curr}')
    
    if err_curr < 1e-4:
        print('Converged!')
        break
    
    # Plot current solution
    if its % 10 == 0:  # Plot every 10 iterations
        plt.clf()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Xint, Yint, Np, cmap='turbo', alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(r'$N_+$')
        plt.pause(0.01)

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