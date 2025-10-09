import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, csr_matrix, bmat
from scipy.sparse.linalg import spsolve, gmres, LinearOperator
from sksparse.cholmod import cholesky

# Delta functions
def delta_a(r, a):
    return (1/(2*np.pi*a**2)) * np.exp(-0.5*(r/a)**2)

def delta_r(r, dx):
    r = abs(r/dx)
    if r < 1:
        return (3 - 2*r + np.sqrt(1 + 4*r - 4*r**2)) / 8 / dx
    elif r < 2:
        return (5 - 2*r - np.sqrt(-7 + 12*r - 4*r**2)) / 8 / dx
    else:
        return 0.0

def spreadQ(X, Y, xq, yq, q, delta, cut):
    Ny, Nx = X.shape
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    Sq = np.zeros_like(X)

    for k in range(len(q)):
        xk, yk, qk = xq[k], yq[k], q[k]

        # Determine the grid region influenced by this point
        i_min = max(int((xk - cut - X[0,0]) / dx), 0)
        i_max = min(int((xk + cut - X[0,0]) / dx) + 1, Nx)
        j_min = max(int((yk - cut - Y[0,0]) / dy), 0)
        j_max = min(int((yk + cut - Y[0,0]) / dy) + 1, Ny)

        # Spread contribution only to local patch
        for j in range(j_min, j_max):
            for i in range(i_min, i_max):
                dxk = X[j, i] - xk
                dyk = Y[j, i] - yk
                R = np.sqrt(dxk**2 + dyk**2)

                if R <= cut:
                    Sq[j, i] += qk * delta(R, dx) * dx * dy

    # Flatten result in Fortran order (column-major)
    return Sq.flatten(order='F')


def interpPhi(X, Y, xq, yq, Phi, delta, cut):
    Ny, Nx = X.shape
    Phi_reshaped = Phi.reshape(Ny, Nx, order='F')
    Jphi = np.zeros_like(xq)
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    for k in range(len(xq)):
        xk, yk = xq[k], yq[k]

        # Determine local grid patch indices within delta support
        i_min = max(int((xk - cut - X[0,0]) / dx), 0)
        i_max = min(int((xk + cut - X[0,0]) / dx) + 1, Nx)
        j_min = max(int((yk - cut - Y[0,0]) / dy), 0)
        j_max = min(int((yk + cut - Y[0,0]) / dy) + 1, Ny)

        # Accumulate weighted sum over the local region
        s = 0.0
        for j in range(j_min, j_max):
            for i in range(i_min, i_max):
                dxk = X[j, i] - xk
                dyk = Y[j, i] - yk
                R = np.sqrt(dxk**2 + dyk**2)

                if R <= cut:
                    s += Phi_reshaped[j, i] * delta(R, dx)

        Jphi[k] = dx * dy * s

    return Jphi

def AxLinearOperator(shape, UGridX, UGridY, VGridX, VGridY, xib, yib, delta, cut, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y):
    n = shape
    def mv(unknowns):
        return apply_A(unknowns, UGridX, UGridY, VGridX, VGridY, xib, yib, delta, cut, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y)
    return LinearOperator((n, n), matvec=mv)

def apply_A(x, UGridX, UGridY, VGridX, VGridY, xib, yib, delta, cut, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y):
    U = x[0:N_U]
    offset = N_U
    V = x[offset:offset + N_V]
    offset = offset + N_V
    P = x[offset:offset + N_P]
    offset = offset + N_P
    Lam_X = x[offset:offset + Nib]
    offset = offset + Nib
    Lam_Y = x[offset:offset + Nib]

    res = np.zeros_like(x)
    res[0:N_U] = Lap_U @ U - G_x @ P + spreadQ(UGridX, UGridY, xib, yib, Lam_X, delta, cut)
    offset = N_U
    res[offset:offset + N_V] = Lap_V @ V - G_y @ P + spreadQ(VGridX, VGridY, xib, yib, Lam_Y, delta, cut)
    offset = offset + N_V
    res[offset:offset + N_P] = D_x @ U + D_y @ V
    offset = offset + N_P
    res[offset:offset + Nib] = interpPhi(UGridX, UGridY, xib, yib, U, delta, cut)
    offset = offset + Nib
    res[offset:offset + Nib] = interpPhi(VGridX, VGridY, xib, yib, V, delta, cut)
    
    return res

L = 1.0
rad = 0.25
size = 4
tol = 1e-4

size_range = range(size)
N = np.zeros(size)
Nib_array = np.zeros(size)
for s in size_range:
    N[s] = 15*(size_range[s] + 3)

# compute Nib_max
Nx_max = int(N[-1])
x_max = np.linspace(0, L, Nx_max + 1)
dx_max = x_max[1] - x_max[0]
dth_max = dx_max / rad
theta_max = np.arange(0, 2*np.pi - dth_max, dth_max)
Nib_max = len(theta_max)

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
Lam_X_NumericalList = np.zeros((int(Nib_max), size))
Lam_X_ExactList = np.ones((int(Nib_max), size))
Lam_Y_NumericalList = np.zeros((int(Nib_max), size))
Lam_Y_ExactList = np.ones((int(Nib_max), size))

for k in size_range:
    # Parameters
    Nx = int(N[k])
    Ny = Nx
    x = np.linspace(0, L, Nx + 1)
    dx = x[1] - x[0]
    dx2 = dx**2
    y = x.copy()
    dy = dx

    cut = 6 * 1.2 * dx # cutoff value

    # IB
    dth = dx / rad
    theta = np.arange(0, 2*np.pi - dth, dth)
    Nib = len(theta)
    Nib_array[k] = Nib
    xib = 0.5 + rad * np.cos(theta) 
    yib = 0.5 + rad * np.sin(theta)
    n_x = np.cos(theta)
    n_y = np.sin(theta)

    x_trunc = x[:-1]    # length Nx
    y_trunc = y[:-1]    # length Ny
    x_mid = x_trunc + dx / 2
    y_mid = y_trunc + dy / 2
    y_offset = y_trunc + dy
    y_offset_trunc = y_offset[1:]

    UGridX, UGridY = np.meshgrid(x_trunc, y_mid)
    VGridX, VGridY = np.meshgrid(x_mid, y_offset_trunc)

    N_U = Nx * Ny
    Ny_minus = Ny - 1
    N_V = Nx * Ny_minus
    N_P = Nx * Ny

    # RHS
    f = np.zeros((Ny, Nx))
    g = np.zeros((Ny_minus, Nx))
    h = np.zeros((Ny, Nx))
    z_x = np.zeros(Nib)
    z_y = np.zeros(Nib)

    # Analytic solutions
    UExact = np.zeros((Ny, Nx))
    VExact = np.zeros((Ny_minus, Nx))
    PExact = np.zeros((Ny, Nx))
    lam_x_exact = interpPhi(UGridX, UGridY, xib, yib, UExact, delta_r, cut)
    lam_y_exact = interpPhi(VGridX, VGridY, xib, yib, VExact, delta_r, cut)

    exact_sol = np.concatenate([UExact.ravel(order='F'), VExact.ravel(order='F'), PExact.ravel(order='F'), lam_x_exact, lam_y_exact])

    def compute_f(xx, yy):
        return -8 * np.pi**2 * np.sin(2*np.pi*xx) * np.sin(2*np.pi*yy) + 2 * np.pi * np.sin(2*np.pi*xx) * np.sin(2*np.pi*yy)

    def compute_g(xx, yy):
        #return -4 * np.pi**2 * np.cos(2*np.pi*xx) * (2*np.cos(2*np.pi*yy) - 1) - 2 * np.pi * np.cos(2*np.pi*yy) * np.cos(2*np.pi*xx)
        return -4 * np.pi**2 * np.cos(2*np.pi*xx) * (2 * np.cos(2*np.pi*yy) - 1) - 2 * np.pi * np.cos(2*np.pi*yy) * np.cos(2*np.pi*xx)

    for j in range(Ny):
        for i in range(Nx):
            f[j, i] = compute_f(x_trunc[i], y_mid[j])
            UExact[j, i] = compute_U(x_trunc[i], y_mid[j])
            PExact[j, i] = compute_P(x_mid[i], y_mid[j])

    for j in range(Ny_minus):
        for i in range(Nx):
            g[j, i] = compute_g(x_mid[i], y_offset_trunc[j])
            VExact[j, i] = compute_V(x_mid[i], y_offset_trunc[j])

    z_x[:] = interpPhi(UGridX, UGridY, xib, yib, UExact, delta_r, cut)
    z_y[:] = interpPhi(VGridX, VGridY, xib, yib, VExact, delta_r, cut)

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
    #dLap_U = cholesky(Lap_U) 
    #dLap_V = cholesky(Lap_V) 

    # Gradients
    Dx_b = diags([np.ones(Nx), -np.ones(Nx)], offsets=[0, -1], shape=(Nx, Nx), format='lil')
    Dx_b[0, -1] = -1.0  # periodic: U[0] uses P[0] - P[Nx-1]
    D_x_backward = (Dx_b / dx).tocsr()
    G_x = kron(D_x_backward, eye(Ny, format='csr'), format='csr')

    D_y_small = diags([-np.ones(Ny_minus), np.ones(Ny_minus)], offsets=[0, 1], shape=(Ny_minus, Ny), format='csr') / dy
    G_y = kron(eye(Nx, format='csr'), D_y_small, format='csr')

    # Divergence (note signs)
    D_x = -G_x.transpose()
    D_y = -G_y.transpose()

    # Zero blocks for bmat (explicit shapes)
    # Z_UV = csr_matrix((N_U, N_V))
    # Z_UP = csr_matrix((N_U, N_P))
    # Z_VU = csr_matrix((N_V, N_U))
    # Z_VP = csr_matrix((N_V, N_P))
    # Z_PU = csr_matrix((N_P, N_U))
    # Z_PV = csr_matrix((N_P, N_V))
    # Z_PNib = csr_matrix((N_P, Nib))
    # Z_NibP = csr_matrix((Nib, N_P))
    # Z_NibNib = csr_matrix((Nib, Nib))
    # Z_UNib = csr_matrix((N_U, Nib))
    # Z_NibU = csr_matrix((Nib, N_U))
    # Z_VNib = csr_matrix((N_V, Nib))
    # Z_NibV = csr_matrix((Nib, N_V))

    # S_U = np.zeros((N_U,Nib))
    # S_V = np.zeros((N_V,Nib))
    # J_U = np.zeros((Nib,N_U))
    # J_V = np.zeros((Nib,N_V))

    # # build dense S matrices
    # for ib in range(Nib):
    #     one_hot = np.zeros(Nib)
    #     one_hot[ib] = 1
    #     S_U[:,ib] = spreadQ(UGridX, UGridY, xib, yib, one_hot, delta_r, cut)
    #     S_V[:,ib] = spreadQ(VGridX, VGridY, xib, yib, one_hot, delta_r, cut)

    # # build dense J matrices
    # for it in range(N_U):
    #     one_hot = np.zeros(N_U)
    #     one_hot[it] = 1
    #     J_U[:,it] = interpPhi(UGridX, UGridY, xib, yib, one_hot, delta_r, cut)

    # for it in range(N_V):
    #     one_hot = np.zeros(N_V)
    #     one_hot[it] = 1
    #     J_V[:,it] = interpPhi(VGridX, VGridY, xib, yib, one_hot, delta_r, cut)

    # Saddle point system
    # A = bmat([
    #     [Lap_U,                  Z_UV,            -G_x,             S_U,            Z_UNib],
    #     [Z_VU,                   Lap_V,           -G_y,             Z_VNib,         S_V],
    #     [D_x,                    D_y,             Z_PU,             Z_PNib,         Z_PNib],
    #     [J_U,                    Z_NibV,          Z_NibP,           Z_NibNib,       Z_NibNib], 
    #     [Z_NibU,                 J_V,             Z_NibP,           Z_NibNib,       Z_NibNib] 
    # ], format='csr')

    # RHS vector (Fortran order flattening already done)
    RHS = np.concatenate([f_bc, g_bc, h_bc, z_x, z_y])
    #RHS = np.concatenate([f_bc, g_bc, h_bc, z_x, z_y])

    # Solve
    shape = N_U + N_V + N_P + 2 * Nib
    AxOp = AxLinearOperator(shape, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_r, cut, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y)
    sol, info = gmres(AxOp, RHS, rtol=tol, maxiter=1000, restart=500, x0=exact_sol, callback=lambda rk: print(f"GMRES residual: {np.linalg.norm(rk)}"))

    #sol = spsolve(A, RHS)
    # Split (no change in ordering of partition)
    U = sol[:N_U]
    V = sol[N_U:N_U + N_V]
    P = sol[N_U + N_V:N_U + N_V + N_P]
    lam_X = sol[N_U + N_V + N_P:N_U + N_V + N_P + Nib]
    lam_Y = sol[N_U + N_V + N_P + Nib:]

    P = P - np.mean(P)

    # Append to solution array 
    UNumericalList[0:Nx**2, k] = U
    VNumericalList[0:Nx * (Ny - 1), k] = V
    PNumericalList[0:Nx**2, k] = P
    Lam_X_NumericalList[0:Nib, k] = lam_X
    Lam_Y_NumericalList[0:Nib, k] = lam_Y
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
err2Norm_Lam_X = np.zeros(size)
err2Norm_Lam_Y = np.zeros(size)
for i in size_range:
    err2Norm_U[i] = np.linalg.norm(UExactList[0:int(N[i]**2), i] - UNumericalList[0:int(N[i]**2), i], ord=2) / np.linalg.norm(UExactList[0:int(N[i]**2),i], ord=2)
    err2Norm_V[i] = np.linalg.norm(VExactList[0:int(N[i]*(N[i]-1)), i] - VNumericalList[0:int(N[i]*(N[i]-1)), i], ord=2) / np.linalg.norm(VExactList[0:int(N[i]*(N[i]-1)),i], ord=2)
    err2Norm_P[i] = np.linalg.norm(PExactList[0:int(N[i]**2), i] - PNumericalList[0:int(N[i]**2), i], ord=2) / np.linalg.norm(PExactList[0:int(N[i]**2),i], ord=2)
    err2Norm_Lam_X[i] = np.linalg.norm(Lam_X_ExactList[0:int(Nib_array[i]), i] - Lam_X_NumericalList[0:int(Nib_array[i]), i], ord=2) / np.linalg.norm(Lam_X_ExactList[0:int(Nib_array[i]), i], ord=2)
    err2Norm_Lam_Y[i] = np.linalg.norm(Lam_Y_ExactList[0:int(Nib_array[i]), i] - Lam_Y_NumericalList[0:int(Nib_array[i]), i], ord=2) / np.linalg.norm(Lam_Y_ExactList[0:int(Nib_array[i]), i], ord=2)

h = 1.0 / N
def rates(err):
    r = []
    for i in range(1, len(err)):
        r.append(np.log(err[i]/err[i-1]) / np.log(h[i]/h[i-1]))
    return np.array(r)

print("U rates:", rates(err2Norm_U))
print("V rates:", rates(err2Norm_V))
print("P rates:", rates(err2Norm_P))
print("Lam_X rates:", rates(err2Norm_Lam_X))
print("Lam_Y rates:", rates(err2Norm_Lam_Y))

plt.figure(figsize=(6,5))
plt.loglog(N, err2Norm_U, '--o', label=r'$Err_U$')
plt.loglog(N, err2Norm_V, '--o', label=r'$Err_V$')
plt.loglog(N, err2Norm_P, '--o', label=r'$Err_P$')
plt.loglog(N, err2Norm_Lam_X, '--o', label=r'$Err_{\lambda_X}$')
plt.loglog(N, err2Norm_Lam_Y, '--o', label=r'$Err_{\lambda_Y}$')
plt.loglog(N, h**2, '--', label='2nd order')

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
