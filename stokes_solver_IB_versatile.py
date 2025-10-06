import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, csr_matrix, bmat
from scipy.sparse.linalg import spsolve

# Delta functions
def delta_a(r, a):
    return (1/(2*np.pi*a**2)) * np.exp(-0.5*(r/a)**2)

def delta_r(r):
    return (1/(1.2*dx))**2 * r * delta_a(r, 1.2*dx)

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

size = 5
size_range = range(size)
N = np.zeros(size)
for s in size_range:
    N[s] = 32 + 5 * (size_range[s] + 1)

def compute_U(xx, yy):
    return np.sin(2*np.pi*yy) * np.sin(2*np.pi*xx)

def compute_V(xx, yy):
    return np.cos(2*np.pi*xx) * (np.cos(2*np.pi*yy) - 1)

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

    # IB
    rad = 0.25
    dth = dx / rad
    theta = np.arange(0, 2*np.pi - dth, dth)
    Nib = len(theta)
    xib = 0.5 + rad * np.cos(theta) 
    yib = 0.5 + rad * np.sin(theta)
    n_x = np.cos(theta)
    n_y = np.sin(theta)

    xint = x[:-1]    # length Nx
    yint = y[:-1]    # length Ny
    y_j_U = yint + dx / 2
    x_i_V = xint + dy / 2
    y_j_offset = yint + dy
    y_j_V = y_j_offset[1:]

    UGridX, UGridY = np.meshgrid(xint, y_j_U)
    VGridX, VGridY = np.meshgrid(x_i_V, y_j_V)

    N_U = Nx * Ny
    Ny_minus = Ny - 1
    N_V = Nx * Ny_minus
    N_P = Nx * Ny

    # RHS
    f = np.zeros((Ny, Nx))
    g = np.zeros((Ny_minus, Nx))
    h = np.zeros((Ny, Nx))
    z = np.zeros(Nib)

    # Analytic solutions
    UExact = np.zeros((Ny, Nx))
    VExact = np.zeros((Ny_minus, Nx))
    PExact = np.zeros((Ny, Nx))
    lam_exact = np.ones(Nib)

    def compute_f(xx, yy):
        return -8 * np.pi**2 * np.sin(2*np.pi*xx) * np.sin(2*np.pi*yy) + 2 * np.pi * np.sin(2*np.pi*xx) * np.sin(2*np.pi*yy)

    def compute_g(xx, yy):
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

    for j in range(Nib):
        z[:] = interpPhi(UGridX, UGridY, xib, yib, UExact, delta_r) + interpPhi(VGridX, VGridY, xib, yib, VExact, delta_r)
        
    f_bc_mat = np.zeros((Ny, Nx))
    g_bc_mat = np.zeros((Ny_minus, Nx))
    h_bc_mat = np.zeros((Ny, Nx))

    f_bc = f.ravel(order='F') + spreadQ(UGridX, UGridY, xib, yib, lam_exact, delta_r) + f_bc_mat.ravel(order='F')
    g_bc = g.ravel(order='F') + spreadQ(VGridX, VGridY, xib, yib, lam_exact, delta_r) + g_bc_mat.ravel(order='F')
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
    e = np.ones(Nx)
    Dx_c = diags([-0.5*e, 0*e, 0.5*e], offsets=[-1,0,1], shape=(Nx, Nx), format='lil')
    Dx_c[0, -1] = -0.5
    Dx_c[-1, 0]  =  0.5
    D_x_centered = (Dx_c / dx).tocsr()
    G_x = kron(D_x_centered, eye(Ny, format='csr'), format='csr')

    #plt.spy(G_x)
    #plt.show()

    ey = np.ones(Ny)
    D_y_small = diags([-0.5*ey, 0.5*ey], offsets=[-1, 1],
                    shape=(Ny_minus, Ny), format='lil')

    D_y_small = (D_y_small / dy).tocsr()

    G_y = kron(eye(Nx, format='csr'), D_y_small, format='csr')

    #plt.spy(G_y)
    #plt.show()

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
    Z_PNib = csr_matrix((N_P, Nib))
    Z_NibP = csr_matrix((Nib, N_P))
    Z_NibNib = csr_matrix((Nib, Nib))

    S_U = np.zeros((N_U,Nib))
    S_V = np.zeros((N_V,Nib))
    J_U = np.zeros((Nib,N_U))
    J_V = np.zeros((Nib,N_V))

    # build dense S matrices
    for ib in range(Nib):
        one_hot = np.zeros(Nib)
        one_hot[ib] = 1
        S_U[:,ib] = spreadQ(UGridX, UGridY, xib, yib, one_hot, delta_r)
        S_V[:,ib] = spreadQ(VGridX, VGridY, xib, yib, one_hot, delta_r)

    # build dense J matrices
    for it in range(N_U):
        one_hot = np.zeros(N_U)
        one_hot[it] = 1
        J_U[:,it] = interpPhi(UGridX, UGridY, xib, yib, one_hot, delta_r)

    for it in range(N_V):
        one_hot = np.zeros(N_V)
        one_hot[it] = 1
        J_V[:,it] = interpPhi(VGridX, VGridY, xib, yib, one_hot, delta_r)

    for j in range(Ny_minus):
        for i in range(Nx):
            g[j, i] = compute_g(x_i_V[i], yint[j] + dy)
            VExact[j, i] = compute_V(x_i_V[i], yint[j] + dy)

    # Saddle point system
    A = bmat([
        [Lap_U,                  Z_UV,            -G_x,             S_U],
        [Z_VU,                   Lap_V,           -G_y,             S_V],
        [D_x,                    D_y,             Z_PU,             Z_PNib],
        [J_U,                    J_V,             Z_NibP,           Z_NibNib] 
    ], format='csr')

    # RHS vector (Fortran order flattening already done)
    RHS = np.concatenate([f_bc, g_bc, h_bc, z])

    # Solve
    sol = spsolve(A, RHS)

    # Split (no change in ordering of partition)
    U = sol[:N_U]
    V = sol[N_U:N_U + N_V]
    P = sol[N_U + N_V:N_U + N_V + N_U]
    lam = sol[N_U + N_V + N_U:]

    #P = P - np.mean(P)

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

    VFull = np.zeros((Ny + 2, Nx + 2))
    VFull[1:Ny, 1:Nx + 1] = Vplot
    VFull[:,0] = VFull[:,1]
    VFull[:,-1] = VFull[:,-2]

# Check errors
err2Norm_U = np.zeros(size)
err2Norm_V = np.zeros(size)
err2Norm_P = np.zeros(size)
second_order = np.zeros(size)
for i in size_range:
    err2Norm_U[i] = np.linalg.norm(UExactList[0:int(N[i]**2), i] - UNumericalList[0:int(N[i]**2), i], ord=2) / np.linalg.norm(UExactList[0:int(N[i]**2),i], ord=2)
    err2Norm_V[i] = np.linalg.norm(VExactList[0:int(N[i]*(N[i]-1)), i] - VNumericalList[0:int(N[i]*(N[i]-1)), i], ord=2) / np.linalg.norm(VExactList[0:int(N[i]*(N[i]-1)),i], ord=2)
    err2Norm_P[i] = np.linalg.norm(PExactList[0:int(N[i]**2), i] - PNumericalList[0:int(N[i]**2), i], ord=2) / np.linalg.norm(PExactList[0:int(N[i]**2),i], ord=2)
    second_order[i] = 10**(-i)

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
plt.loglog(N, second_order, '--o', label=r'$Second-Order$')
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

plt.show()
