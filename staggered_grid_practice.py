import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, csr_matrix, bmat
from scipy.sparse.linalg import spsolve

# Parameters
Nx = 64
Ny = Nx
L = 1.0
x = np.linspace(0, L, Nx + 1)
dx = x[1] - x[0]
dx2 = dx**2
y = x.copy()
dy = dx

xint = x[:-1]    # length Nx
yint = y[:-1]    # length Ny
y_j_U = yint + dx / 2
x_i_V = xint + dy / 2

N_U = Nx * Ny
Ny_minus = Ny - 1
N_V = Nx * Ny_minus
N_P = Nx * Ny

# BCs
V_lower = -3.5
V_upper = -3.5

f_bc_mat = np.zeros((Ny, Nx))
g_bc_mat = np.zeros((Ny_minus, Nx))
h_bc_mat = np.zeros((Ny, Nx))

# IMPORTANT: flatten in Fortran order to mimic MATLAB's column-major ordering
f_bc = f_bc_mat.ravel(order='F')

g_bc_mat[0, :] = V_lower / dy**2
g_bc_mat[-1, :] = V_upper / dy**2
g_bc = g_bc_mat.ravel(order='F')

h_bc_mat[0, :] = -V_lower / dy
h_bc_mat[-1, :] = V_upper / dy
h_bc = h_bc_mat.ravel(order='F')

# Sigma
def compute_sigma(xx, yy):
    return 50 * (np.sin(2*np.pi*xx) * np.sin(4*np.pi*yy) + 1)

sigma_matrix_U = np.zeros((Ny, Nx))
for j in range(Ny):
    for i in range(Nx):
        sigma_matrix_U[j, i] = compute_sigma(xint[i], y_j_U[j])
sigma_vec_U = sigma_matrix_U.ravel(order='F')
sigma_U_diag = diags(sigma_vec_U, 0, format='csr')

sigma_matrix_V = np.zeros((Ny_minus, Nx))
for j in range(Ny_minus):
    for i in range(Nx):
        sigma_matrix_V[j, i] = compute_sigma(x_i_V[i], yint[j] + dy)
sigma_vec_V = sigma_matrix_V.ravel(order='F')
sigma_V_diag = diags(sigma_vec_V, 0, format='csr')

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
Dx_f = diags([-np.ones(Nx), np.ones(Nx)], offsets=[0, 1], shape=(Nx, Nx), format='lil')
Dx_f[-1, 0] = 1.0
D_x_forward = (Dx_f / dx).tocsr()
G_x = kron(D_x_forward, eye(Ny, format='csr'), format='csr')

D_y_small = diags([-np.ones(Ny_minus), np.ones(Ny_minus)], offsets=[0, 1], shape=(Ny_minus, Ny), format='csr') / dy
G_y = kron(eye(Nx, format='csr'), D_y_small, format='csr')

# Divergence (note signs)
D_x = -G_x.transpose()
D_y = -G_y.transpose()

# Zero blocks for bmat (explicit shapes)
Z_UV = csr_matrix((N_U, N_V))
Z_UP = csr_matrix((N_U, N_P))
Z_VU = csr_matrix((N_V, N_U))
Z_VP = csr_matrix((N_V, N_P))
Z_PU = csr_matrix((N_P, N_U))
Z_PV = csr_matrix((N_P, N_V))

# Saddle point system
A = bmat([
    [Lap_U + sigma_U_diag,   Z_UV,            G_x],
    [Z_VU,                   Lap_V + sigma_V_diag, G_y],
    [D_x,                    D_y,             Z_PU.T]  # Z_PU.T is zero but sized correctly
], format='csr')

# RHS vector (Fortran order flattening already done)
RHS = np.concatenate([f_bc, g_bc, h_bc])

# Solve
sol = spsolve(A, RHS)

# Split (no change in ordering of partition)
U = sol[:N_U]
V = sol[N_U:N_U + N_V]
P = sol[N_U + N_V:]

# Reshape back using Fortran order to match MATLAB layout
Uplot = U.reshape((Ny, Nx), order='F')
Vplot = V.reshape((Ny_minus, Nx), order='F')
Pplot = P.reshape((Ny, Nx), order='F')

# Build full arrays with ghost rows/cols similar to MATLAB
UFull = np.full((Ny + 2, Nx + 2), np.nan)
UFull[1:Ny + 1, 1:Nx + 1] = Uplot
UFull[0, :] = UFull[1, :]
UFull[-1, :] = UFull[-2, :]

VFull = V_lower * np.ones((Ny + 2, Nx + 2))
VFull[1:Ny, 1:Nx + 1] = -Vplot
VFull[0, :] = VFull[1, :]
VFull[-1, :] = VFull[-2, :]

xplot = np.linspace(0, L, Nx + 2)
yplot = np.linspace(0, L, Ny + 2)
Xplot, Yplot = np.meshgrid(xplot, yplot)

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

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xplot, Yplot, np.sqrt(UFull**2 + VFull**2), cmap=cmap, edgecolor='none')
ax.set_title("|velocity|")
ax.set_xlabel("x"); ax.set_ylabel("y")

plt.show()
