import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, csr_matrix, bmat
from scipy.sparse.linalg import spsolve

def spreadQ(X, Y, xq, yq, q, delta):
    """
    Spread point values q at positions (xq, yq) to the grid (X,Y) using kernel delta.
    """
    Sq = np.zeros_like(X)
    Nq = len(q)

    for k in range(Nq):
        Rk = np.sqrt((X - xq[k])**2 + (Y - yq[k])**2)
        Sq += q[k] * delta(Rk)

    # flatten with Fortran ordering (column-major, like MATLAB)
    return Sq.flatten(order='F')


def interpPhi(X, Y, xq, yq, Phi, delta):
    """
    Interpolate grid field Phi (given as a flat vector) to point values at (xq, yq).
    """
    # reshape Phi into 2D with Fortran ordering
    nx, ny = X.shape
    Phi = np.reshape(Phi, (nx, ny), order='F')

    Jphi = np.zeros_like(xq, dtype=float)
    Nq = len(xq)
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    for k in range(Nq):
        Rk = np.sqrt((X - xq[k])**2 + (Y - yq[k])**2)
        Jphi[k] = dx * dy * np.sum(Phi * delta(Rk))

    return Jphi

size = 10
size_range = range(size)
N = np.zeros(size)
for s in size_range:
    N[s] = 10*(size_range[s] + 1)

def compute_U(xx, yy):
    return np.sin(2*np.pi*yy) * np.sin(2*np.pi*xx)

def compute_V(xx, yy):
    return -3.5 + np.cos(2*np.pi*xx) * (np.cos(2*np.pi*yy) - 1)

def compute_P(xx, yy):
    return np.sin(2*np.pi*yy) * np.cos(2*np.pi*xx)

UNumericalList = np.zeros((int(N[-1] * N[-1]), size))
UExactList = np.zeros((int(N[-1] * N[-1]), size))
PNumericalList = np.zeros((int(N[-1] * N[-1]), size))
PExactList = np.zeros((int(N[-1] * N[-1]), size))
VNumericalList = np.zeros((int(N[-1] * (N[-1] - 1)), size))
VExactList = np.zeros((int(N[-1] * (N[-1] - 1)), size))

for k in size_range:
    # Parameters
    Nx = int(N[k])
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

    # RHS
    f = np.zeros((Ny, Nx))
    g = np.zeros((Ny_minus, Nx))
    h = np.zeros((Ny, Nx))

    # Analytic solutions
    UExact = np.zeros((Ny, Nx))
    VExact = np.zeros((Ny_minus, Nx))
    PExact = np.zeros((Ny, Nx))

    def compute_f(xx, yy):
        return -8 * np.pi**2 * np.sin(2*np.pi*xx) * np.sin(2*np.pi*yy) + 2 * np.pi * np.sin(2*np.pi*xx) * np.sin(2*np.pi*yy)

    def compute_g(xx, yy):
        #return -4 * np.pi**2 * np.cos(2*np.pi*xx) * (2*np.cos(2*np.pi*yy) - 1) - 2 * np.pi * np.cos(2*np.pi*yy) * np.cos(2*np.pi*xx)
        return -4 * np.pi**2 * np.cos(2*np.pi*xx) * (2 * np.cos(2*np.pi*yy) - 1) - 2 * np.pi * np.cos(2*np.pi*yy) * np.cos(2*np.pi*xx)

    for j in range(Ny):
        for i in range(Nx):
            f[j, i] = compute_f(xint[i], y_j_U[j])
            UExact[j, i] = compute_U(xint[i], y_j_U[j])
            PExact[j, i] = compute_P(xint[i], yint[j])

    for j in range(Ny_minus):
        for i in range(Nx):
            g[j, i] = compute_g(x_i_V[i], yint[j] + dy)
            VExact[j, i] = compute_V(x_i_V[i], yint[j] + dy)

    # BCs
    V_lower = -3.5
    V_upper = -3.5

    f_bc_mat = np.zeros((Ny, Nx))
    g_bc_mat = np.zeros((Ny_minus, Nx))
    h_bc_mat = np.zeros((Ny, Nx))

    # IMPORTANT: flatten in Fortran order to mimic MATLAB's column-major ordering
    f_bc = f.ravel(order='F') + f_bc_mat.ravel(order='F')

    g_bc_mat[0, :] = -V_lower / dy**2
    g_bc_mat[-1, :] = -V_upper / dy**2
    g_bc = g.ravel(order='F') + g_bc_mat.ravel(order='F')

    h_bc_mat[0, :] = V_lower / dy
    h_bc_mat[-1, :] = -V_upper / dy
    h_bc = h_bc_mat.ravel(order='F')

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

    xint[i], y_j_U[j]

    # Saddle point system
    A = bmat([
        [Lap_U,                  Z_UV,            -G_x],
        [Z_VU,                   Lap_V,           -G_y],
        [D_x,                    D_y,             Z_PU.T]
    ], format='csr')

    # RHS vector (Fortran order flattening already done)
    RHS = np.concatenate([f_bc, g_bc, h_bc])

    # Solve
    sol = spsolve(A, RHS)

    # Split (no change in ordering of partition)
    U = sol[:N_U]
    V = sol[N_U:N_U + N_V]
    P = sol[N_U + N_V:]

    P = P - np.mean(P)

    # Append to solution array 
    UNumericalList[0:Nx**2, k] = U
    VNumericalList[0:Nx * (Ny - 1), k] = V
    PNumericalList[0:Nx**2, k] = P
    UExactList[0:Nx**2, k] = UExact.ravel(order='F')
    VExactList[0:Nx * (Ny - 1), k] = VExact.ravel(order='F')
    PExactList[0:Nx**2, k] = PExact.ravel(order='F')

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
    VFull[1:Ny, 1:Nx + 1] = Vplot
    VFull[:,0] = VFull[:,1]
    VFull[:,-1] = VFull[:,-2]

# Check errors
err2Norm_U = np.zeros(size)
err2Norm_V = np.zeros(size)
err2Norm_P = np.zeros(size)
for i in size_range:
    err2Norm_U[i] = np.linalg.norm(UExactList[0:int(N[i]**2), i] - UNumericalList[0:int(N[i]**2), i], ord=2) / np.linalg.norm(UExactList[0:int(N[i]**2),i], ord=2)
    err2Norm_V[i] = np.linalg.norm(VExactList[0:int(N[i]*(N[i]-1)), i] - VNumericalList[0:int(N[i]*(N[i]-1)), i], ord=2) / np.linalg.norm(VExactList[0:int(N[i]*(N[i]-1)),i], ord=2)
    err2Norm_P[i] = np.linalg.norm(PExactList[0:int(N[i]**2), i] - PNumericalList[0:int(N[i]**2), i], ord=2) / np.linalg.norm(PExactList[0:int(N[i]**2),i], ord=2)

h = 1.0 / N
def rates(err):
    r = []
    for i in range(1, len(err)):
        r.append(np.log(err[i]/err[i-1]) / np.log(h[i]/h[i-1]))
    return np.array(r)

print("U rates:", rates(err2Norm_U))
print("V rates:", rates(err2Norm_V))
print("P rates:", rates(err2Norm_P))

plt.figure(figsize=(6,5))
plt.loglog(N, err2Norm_U, '--o', label=r'$Err$')
plt.loglog(N, err2Norm_V, '--o', label=r'$Err$')
plt.loglog(N, err2Norm_P, '--o', label=r'$Err$')

plt.xlabel(r'$N$', fontsize=22)
plt.ylabel(r'$Err$', fontsize=22)
plt.title(r'Relative error of $u_n(x, y)$, $p=2$ norm', fontsize=22)
plt.legend(fontsize=22, loc='upper right')  # matplotlib uses "upper right"

plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.show()

xplot = np.linspace(0, L, Nx + 2)
yplot = np.linspace(0, L, Ny + 2)
Xplot, Yplot = np.meshgrid(xplot, yplot)
Xint, Yint = np.meshgrid(xint, yint)

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
