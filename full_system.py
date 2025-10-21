
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit
from sksparse.cholmod import cholesky
from scipy.sparse import spdiags, eye, kron, diags
from scipy.sparse.linalg import spsolve, gmres, LinearOperator
from scipy.linalg import qr, lstsq
from scipy.io import loadmat, savemat
from scipy.interpolate import Akima1DInterpolator, interpn
import CPEO_utils as cpeo

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

D_y_small = diags([-np.ones(Ny), np.ones(Ny)], offsets=[0, 1], shape=(Ny, Ny), format='csr') / dy
D_x_small = diags([-np.ones(Nx), np.ones(Nx)], offsets=[0, 1], shape=(Nx, Nx), format='csr') / dx
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
    return cpeo.Build_RHS_Schur_System(ctxt, ctxt_BCs_Schur, U_fluid, V_fluid, G_x, G_y, Lap, dLap, G_d_G, delta_layer, Nx, Ny, Nib, Jop, Jop_prime)

def b_Op(ctxt, U_fluid, V_fluid):
    return cpeo.Build_RHS_rho(ctxt, ctxt_BCs, U_fluid, V_fluid, G_x, G_y, Lap, dLap, G_d_G, delta_layer, Nx, Ny, Nib, Jop, Jop_prime)

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
    Phi = u_next[:Ny*Nx].reshape(Ny, Nx)
    Np = u_next[Ny*Nx:2*Nx*Ny].reshape(Ny, Nx)
    Nm = u_next[2*Ny*Nx:3*Nx*Ny].reshape(Ny, Nx)
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

ctxt_R = u_next.copy()

# verify residudal
residual_check_RHS = b_Op(u_next, U_fluid, V_fluid)
residual_check_AxOp = AxOp(u_next)
residual_check = np.linalg.norm(residual_check_AxOp - residual_check_RHS) / np.linalg.norm(residual_check_RHS)

# compute Lambda^E

LambdaE_x, LambdaE_y = cpeo.compute_surface_maxwell_stress(Phi.ravel(order='F'), G_x, G_y, Nx, Ny, Nib, xib, yib, 0, 0, Jop)

################################
#####  solve N*u = F(phi)  #####
################################

# Parameters
x = np.linspace(-L/2, L/2, Nx + 1)
dx = x[1] - x[0]
dx2 = dx**2
y = x.copy()
dy = dx

#tol = 0.01 * dx**2

cut = 6 * 1.2 * dx # cutoff value

delta_x, delta_y = cpeo.make_composite_deltas(dx, n=3)

x_trunc = x[:-1]    # length Nx
y_trunc = y[:-1]    # length Ny
x_mid = x_trunc + dx / 2
y_mid = y_trunc + dy / 2
y_offset = y_trunc[1:]

UGridX, UGridY = np.meshgrid(x_trunc, y_mid)
VGridX, VGridY = np.meshgrid(x_mid, y_offset)

N_U = Nx * Ny
Ny_minus = Ny - 1
N_V = Nx * Ny_minus
N_P = Nx * Ny

# BCs
V_lower = 0
V_upper = 0

# RHS
f = -cpeo.spreadQ_x(UGridX, UGridY, xib, yib, LambdaE_x, delta_x)
g = -cpeo.spreadQ_y(VGridX, VGridY, xib, yib, LambdaE_y, delta_y)
h = np.zeros((Ny, Nx))
z_x = np.zeros(Nib)
z_y = np.zeros(Nib)

f_bc_mat = np.zeros((Ny, Nx))
g_bc_mat = np.zeros((Ny_minus, Nx))
h_bc_mat = np.zeros((Ny, Nx))

f_bc = f.ravel(order='F') + f_bc_mat.ravel(order='F')

g_bc_mat[0, :] = -V_lower / dy**2
g_bc_mat[-1, :] = -V_upper / dy**2
g_bc = g.ravel(order='F') + g_bc_mat.ravel(order='F')

h_bc_mat[0, :] = V_lower / dy
h_bc_mat[-1, :] = -V_upper / dy
h_bc = h.ravel(order='F') + h_bc_mat.ravel(order='F')

# 1-D operators
# x-direction (periodic)
e = np.ones(Nx)
D2_x = diags([e, -2*e, e], offsets=[-1, 0, 1], shape=(Nx, Nx), format='lil')
D2_x[0, -1] = 1.0
D2_x[-1, 0] = 1.0
D2_x = (D2_x / dx2).tocsr()

# y-direction for U (includes ghost treatment -> -3 on boundaries)
ey = np.ones(Ny)
D2_y_U = diags([ey, -2*ey, ey], offsets=[-1, 0, 1], shape=(Ny, Ny), format='lil')
D2_y_U[0, 0] = -3.0
D2_y_U[-1, -1] = -3.0
D2_y_U = (D2_y_U / dy**2).tocsr()

# y-direction for V (interior Ny-1 points)
ev = np.ones(Ny_minus)
D2_y_V = diags([ev, -2*ev, ev], offsets=[-1, 0, 1], shape=(Ny_minus, Ny_minus), format='csr') / dy**2

# Laplacians (Kronecker products)
Lap_U = kron(D2_x, eye(Ny, format='csr'), format='csr') + kron(eye(Nx, format='csr'), D2_y_U, format='csr')
Lap_V = kron(D2_x, eye(Ny_minus, format='csr'), format='csr') + kron(eye(Nx, format='csr'), D2_y_V, format='csr')

# Gradients
Dx_b = diags([np.ones(Nx), -np.ones(Nx)], offsets=[0, -1], shape=(Nx, Nx), format='lil')
Dx_b[0, -1] = -1.0  # periodic: U[0] uses P[0] - P[Nx-1]
D_x_backward = (Dx_b / dx).tocsr()
G_x_staggered = kron(D_x_backward, eye(Ny, format='csr'), format='csr')

D_y_small = diags([-np.ones(Ny_minus), np.ones(Ny_minus)], offsets=[0, 1], shape=(Ny_minus, Ny), format='csr') / dy
G_y_staggered = kron(eye(Nx, format='csr'), D_y_small, format='csr')

# Divergence (note signs)
D_x = -G_x_staggered.transpose()
D_y = -G_y_staggered.transpose()

RHS = np.concatenate([f_bc, g_bc, h_bc, z_x, z_y])
RHS_schur = cpeo.schur_rhs_N(RHS, Lap_U, Lap_V, D_x, D_y, delta_x, delta_y, UGridX, UGridY, VGridX, VGridY, xib, yib, N_U, N_V, N_P, Nib)

# Solve
shape = N_P + 2 * Nib
SchurOp = cpeo.SchurLinearOperator_N(shape, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, Lap_U, Lap_V, G_x_staggered, G_y_staggered, D_x, D_y, N_P, Nib)
sol, info = gmres(SchurOp, RHS_schur, rtol=tol, restart=500, callback=lambda rk: print(f"GMRES residual: {np.linalg.norm(rk)}"))

# Split (no change in ordering of partition)
P = sol[:N_P]
lam_X = sol[N_P:N_P + Nib]
lam_Y = sol[N_P + Nib:]

P = P - np.mean(P)

# Postprocessing: compute U and V
U = cpeo.compute_U_postprocessing(P, lam_X, UGridX, UGridY, xib, yib, Lap_U, G_x_staggered, delta_x, f_bc)
V = cpeo.compute_V_postprocessing(P, lam_Y, VGridX, VGridY, xib, yib, Lap_V, G_y_staggered, delta_y, g_bc)

# Reshape back using Fortran order to match MATLAB layout
Uplot = U.reshape((Ny, Nx), order='F')
Vplot = V.reshape((Ny_minus, Nx), order='F')
Pplot = P.reshape((Ny, Nx), order='F')

# Build full arrays with ghost rows/cols similar to MATLAB
UFull = np.full((Ny + 2, Nx + 2), np.nan)
UFull[1:Ny + 1, 1:Nx + 1] = Uplot
UFull[0, :] = UFull[1, :]
UFull[-1, :] = UFull[-2, :]

PFull = np.full((Ny + 2, Nx + 2), np.nan)
PFull[1:Ny + 1, 1:Nx + 1] = Pplot
PFull[0, :] = UFull[1, :]
PFull[-1, :] = UFull[-2, :]

VFull = V_lower * np.ones((Ny + 2, Nx + 2))
VFull[1:Ny, 1:Nx + 1] = Vplot
VFull[:,0] = VFull[:,1]
VFull[:,-1] = VFull[:,-2]

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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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

U_fluid = (UFull[1:Nx+1,1:Ny+1]).ravel(order='F')
V_fluid = (VFull[1:Nx+1,1:Ny+1]).ravel(order='F')

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
        Phi = u_next[:Ny*Nx].reshape(Ny, Nx)
        Np = u_next[Ny*Nx:2*Nx*Ny].reshape(Ny, Nx)
        Nm = u_next[2*Ny*Nx:3*Nx*Ny].reshape(Ny, Nx)
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
            print('Rphi = rho Converged!')
            break

        # compute Lambda^E

    LambdaE_x, LambdaE_y = cpeo.compute_surface_maxwell_stress(Phi.ravel(order='F'), G_x, G_y, Nx, Ny, Nib, xib, yib, 0, 0, Jop)

    ################################
    #####  solve N*u = F(phi)  #####
    ################################

    # RHS
    f = -cpeo.spreadQ_x(UGridX, UGridY, xib, yib, LambdaE_x, delta_x)
    g = -cpeo.spreadQ_y(VGridX, VGridY, xib, yib, LambdaE_y, delta_y)

    f_bc = f.ravel(order='F') + f_bc_mat.ravel(order='F')
    g_bc = g.ravel(order='F') + g_bc_mat.ravel(order='F')
    h_bc = h.ravel(order='F') + h_bc_mat.ravel(order='F')

    RHS = np.concatenate([f_bc, g_bc, h_bc, z_x, z_y])
    RHS_schur = cpeo.schur_rhs_N(RHS, Lap_U, Lap_V, D_x, D_y, delta_x, delta_y, UGridX, UGridY, VGridX, VGridY, xib, yib, N_U, N_V, N_P, Nib)

    # Solve
    shape = N_P + 2 * Nib
    SchurOp = cpeo.SchurLinearOperator_N(shape, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, Lap_U, Lap_V, G_x_staggered, G_y_staggered, D_x, D_y, N_P, Nib)
    sol, info = gmres(SchurOp, RHS_schur, rtol=tol, restart=500, callback=lambda rk: print(f"GMRES residual: {np.linalg.norm(rk)}"))

    # Split (no change in ordering of partition)
    P = sol[:N_P]
    lam_X = sol[N_P:N_P + Nib]
    lam_Y = sol[N_P + Nib:]

    P = P - np.mean(P)

    # Postprocessing: compute U and V
    U = cpeo.compute_U_postprocessing(P, lam_X, UGridX, UGridY, xib, yib, Lap_U, G_x_staggered, delta_x, f_bc)
    V = cpeo.compute_V_postprocessing(P, lam_Y, VGridX, VGridY, xib, yib, Lap_V, G_y_staggered, delta_y, g_bc)

    # Reshape back using Fortran order to match MATLAB layout
    Uplot = U.reshape((Ny, Nx), order='F')
    Vplot = V.reshape((Ny_minus, Nx), order='F')
    Pplot = P.reshape((Ny, Nx), order='F')

    # Build full arrays with ghost rows/cols similar to MATLAB
    UFull = np.full((Ny + 2, Nx + 2), np.nan)
    UFull[1:Ny + 1, 1:Nx + 1] = Uplot
    UFull[0, :] = UFull[1, :]
    UFull[-1, :] = UFull[-2, :]

    PFull = np.full((Ny + 2, Nx + 2), np.nan)
    PFull[1:Ny + 1, 1:Nx + 1] = Pplot
    PFull[0, :] = UFull[1, :]
    PFull[-1, :] = UFull[-2, :]

    VFull = V_lower * np.ones((Ny + 2, Nx + 2))
    VFull[1:Ny, 1:Nx + 1] = Vplot
    VFull[:,0] = VFull[:,1]
    VFull[:,-1] = VFull[:,-2]

    U_fluid = (UFull[1:Nx+1,1:Ny+1]).ravel(order='F')
    V_fluid = (VFull[1:Nx+1,1:Ny+1]).ravel(order='F')

    residual_check_RHS = b_Op(u_next, U_fluid, V_fluid)
    residual_check_AxOp = AxOp(u_next)
    residual_check = np.linalg.norm(residual_check_AxOp - residual_check_RHS) / np.linalg.norm(residual_check_RHS)
    print(f'New residual = {residual_check}')

    if residual_check < 1e-4:
        print('Full system converged!')
        break

# check full residual 
u_tilde = np.concatenate([U, V, P, lam_X, lam_Y])
Nu = cpeo.apply_N(u_tilde, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x_staggered, G_y_staggered, D_x, D_y)
residual_check_N = np.linalg.norm(Nu - RHS) / np.linalg.norm(RHS)
print(f'New residual Nu = {residual_check}')

plt.figure(figsize=(8, 6))
plt.streamplot(Xplot, Yplot, UFull, VFull, color='red', density=10, linewidth=1, arrowsize=1.5)
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Flow Field Streamline Plot')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(True)
plt.show()
