import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, csr_matrix, bmat
from scipy.sparse.linalg import spsolve, gmres, LinearOperator
from scipy.interpolate import BSpline

import numpy as np
from scipy.interpolate import BSpline

import numpy as np
from scipy.interpolate import BSpline

import numpy as np
from scipy.interpolate import BSpline

def bspline_kernel(n, dx, normalize=True):
    """
    Return a centered, compactly supported B-spline kernel φ_h^n(r)
    such that ∫ φ_h^n(r) dr ≈ 1 (if normalize=True).

    This version treats values outside the knot range as zero (replacing NaNs)
    and optionally enforces numerical normalization.
    """
    k = n

    # Create a centered open-uniform knot vector long enough for degree k.
    # We pick knots spanning [-k-1, k+1] (integer spacing) which gives
    # sufficient support for a centered basis.
    t = np.arange(-(k + 1), (k + 2), 1.0)   # length = 2k + 3 (safe)
    c = np.zeros(len(t) - k - 1)
    c[len(c) // 2] = 1.0

    base = BSpline(t, c, k, extrapolate=False)

    # Precompute a normalization constant (numerical) if requested.
    norm = 1.0
    if normalize:
        # integrate on a range large enough to capture full support:
        R = (k + 1) * dx  # support roughly within +- (k+1)*dx
        rgrid = np.linspace(-R, R, max(401, int(8*(k+1))))
        vals = base(rgrid / dx)
        vals = np.nan_to_num(vals, nan=0.0)   # outside support -> 0
        integral = np.trapz(vals, rgrid)      # integral of φ^n (unscaled)
        # φ_h(r) = (1/dx) * φ^n(r/dx) so integral over r should be integral/dx
        # we want integral/dx == 1 -> normalization factor = 1 / (integral/dx) = dx / integral
        if integral == 0.0:
            raise RuntimeError("B-spline base integral is zero (unexpected).")
        norm = dx / integral

    def phi_h(r):
        # Evaluate base at r/dx, replace NaNs with 0 (outside support)
        vals = base(r / dx)
        vals = np.nan_to_num(vals, nan=0.0)
        return (1.0 / dx) * norm * vals

    return phi_h


def make_composite_deltas(dx, n=3):
    """Return delta_x, delta_y composite kernels for staggered grids."""
    phi_n = bspline_kernel(n, dx, normalize=True)
    phi_np1 = bspline_kernel(n + 1, dx, normalize=True)

    def delta_x(X, Y, x_l, y_l):
        return phi_np1(X - x_l) * phi_n(Y - y_l)

    def delta_y(X, Y, x_l, y_l):
        return phi_n(X - x_l) * phi_np1(Y - y_l)

    return delta_x, delta_y

def interpPhi_x(Xu, Yu, xq, yq, U, delta_x):
    """
    Interpolate u(x,y) from staggered x-face grid (Xu, Yu)
    to Lagrangian points (xq, yq) using composite δ_x.
    """
    if U.ndim == 1:
        U = U.reshape(Xu.shape, order='F')

    Jphi = np.zeros_like(xq)
    dx = Xu[0, 1] - Xu[0, 0]
    dy = Yu[1, 0] - Yu[0, 0]

    for k in range(len(xq)):
        phi = delta_x(Xu, Yu, xq[k], yq[k])
        Jphi[k] = dx * dy * np.sum(U * phi)

    return Jphi

def interpPhi_y(Xv, Yv, xq, yq, V, delta_y):
    """
    Interpolate v(x,y) from staggered y-face grid (Xv, Yv)
    to Lagrangian points (xq, yq) using composite δ_y.
    """
    if V.ndim == 1:
        V = V.reshape(Xv.shape, order='F')

    Jphi = np.zeros_like(xq)
    dx = Xv[0, 1] - Xv[0, 0]
    dy = Yv[1, 0] - Yv[0, 0]

    for k in range(len(xq)):
        phi = delta_y(Xv, Yv, xq[k], yq[k])
        Jphi[k] = dx * dy * np.sum(V * phi)

    return Jphi

def spreadQ_x(Xu, Yu, xq, yq, qx, delta_x):
    """
    Spread Lagrangian x-forces qx (force per unit length)
    to Eulerian x-face grid (Xu, Yu) using composite δ_x.
    """
    Fx = np.zeros_like(Xu)

    xp = np.asarray(xq)
    yp = np.asarray(yq)
    dxs = np.sqrt(np.diff(np.concatenate([xp, xp[:1]]))**2 +
                np.diff(np.concatenate([yp, yp[:1]]))**2)
    ds = np.mean(dxs)

    for k in range(len(qx)):
        Fx += qx[k] * delta_x(Xu, Yu, xq[k], yq[k]) * ds

    return Fx.ravel(order='F')


def spreadQ_y(Xv, Yv, xq, yq, qy, delta_y):
    """
    Spread Lagrangian y-forces qy (force per unit length)
    to Eulerian y-face grid (Xv, Yv) using composite δ_y.
    """
    Fy = np.zeros_like(Xv)

    xp = np.asarray(xq)
    yp = np.asarray(yq)
    dxs = np.sqrt(np.diff(np.concatenate([xp, xp[:1]]))**2 +
                np.diff(np.concatenate([yp, yp[:1]]))**2)
    ds = np.mean(dxs)

    for k in range(len(qy)):
        Fy += qy[k] * delta_y(Xv, Yv, xq[k], yq[k]) * ds

    return Fy.ravel(order='F')


def AxLinearOperator(shape, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y):
    n = shape
    def mv(unknowns):
        return apply_A(unknowns, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y)
    return LinearOperator((n, n), matvec=mv)

def apply_A(x, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y):
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
    res[0:N_U] = Lap_U @ U - G_x @ P + spreadQ_x(UGridX, UGridY, xib, yib, Lam_X, delta_x)
    offset = N_U
    res[offset:offset + N_V] = Lap_V @ V - G_y @ P + spreadQ_y(VGridX, VGridY, xib, yib, Lam_Y, delta_y)
    offset = offset + N_V
    res[offset:offset + N_P] = D_x @ U + D_y @ V
    offset = offset + N_P
    res[offset:offset + Nib] = interpPhi_x(UGridX, UGridY, xib, yib, U, delta_x)
    offset = offset + Nib
    res[offset:offset + Nib] = interpPhi_y(VGridX, VGridY, xib, yib, V, delta_y)
    
    return res

L = 1.0
rad = 0.25
size = 7

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
Lam_X_ExactList = np.zeros((int(Nib_max), size))
Lam_Y_NumericalList = np.zeros((int(Nib_max), size))
Lam_Y_ExactList = np.zeros((int(Nib_max), size))

for k in size_range:
    # Parameters
    Nx = int(N[k])
    Ny = Nx
    x = np.linspace(0, L, Nx + 1)
    dx = x[1] - x[0]
    dx2 = dx**2
    y = x.copy()
    dy = dx

    tol = 0.01 * dx**2

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

    delta_x, delta_y = make_composite_deltas(dx, n=3)

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
    lam_x_exact = np.ones(Nib)
    lam_y_exact = np.ones(Nib)

    def compute_f(xx, yy):
        return -8 * np.pi**2 * np.sin(2*np.pi*xx) * np.sin(2*np.pi*yy) + 2 * np.pi * np.sin(2*np.pi*xx) * np.sin(2*np.pi*yy)

    def compute_g(xx, yy):
        return -4 * np.pi**2 * np.cos(2*np.pi*xx) * (2 * np.cos(2*np.pi*yy) - 1) - 2 * np.pi * np.cos(2*np.pi*yy) * np.cos(2*np.pi*xx)

    for j in range(Ny):
        for i in range(Nx):
            f[j, i] = compute_f(x_trunc[i], y_mid[j])
            UExact[j, i] = compute_U(x_trunc[i], y_mid[j])
            PExact[j, i] = compute_P(x_mid[i], y_mid[j])

    for j in range(Ny_minus):
        for i in range(Nx):
            g[j, i] = compute_g(x_mid[i], y_offset[j])
            VExact[j, i] = compute_V(x_mid[i], y_offset[j])

    exact_sol = np.concatenate([UExact.ravel(order='F'), VExact.ravel(order='F'), PExact.ravel(order='F'), lam_x_exact, lam_y_exact])

    z_x[:] = interpPhi_x(UGridX, UGridY, xib, yib, UExact, delta_x)
    z_y[:] = interpPhi_y(VGridX, VGridY, xib, yib, VExact, delta_y)

    # BCs
    V_lower = -3.5
    V_upper = -3.5

    f_bc_mat = np.zeros((Ny, Nx))
    g_bc_mat = np.zeros((Ny_minus, Nx))
    h_bc_mat = np.zeros((Ny, Nx))

    # IMPORTANT: flatten in Fortran order to mimic MATLAB's column-major ordering
    f_bc = f.ravel(order='F') + spreadQ_x(UGridX, UGridY, xib, yib, lam_x_exact, delta_x) + f_bc_mat.ravel(order='F')

    g_bc_mat[0, :] = -V_lower / dy**2
    g_bc_mat[-1, :] = -V_upper / dy**2
    g_bc = g.ravel(order='F') + spreadQ_y(VGridX, VGridY, xib, yib, lam_y_exact, delta_y) + g_bc_mat.ravel(order='F')

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
    Dx_b = diags([np.ones(Nx), -np.ones(Nx)], offsets=[0, -1], shape=(Nx, Nx), format='lil')
    Dx_b[0, -1] = -1.0  # periodic: U[0] uses P[0] - P[Nx-1]
    D_x_backward = (Dx_b / dx).tocsr()
    G_x = kron(D_x_backward, eye(Ny, format='csr'), format='csr')

    D_y_small = diags([-np.ones(Ny_minus), np.ones(Ny_minus)], offsets=[0, 1], shape=(Ny_minus, Ny), format='csr') / dy
    G_y = kron(eye(Nx, format='csr'), D_y_small, format='csr')

    # Divergence (note signs)
    D_x = -G_x.transpose()
    D_y = -G_y.transpose()

    RHS = np.concatenate([f_bc, g_bc, h_bc, z_x, z_y])

    # Solve
    shape = N_U + N_V + N_P + 2 * Nib
    AxOp = AxLinearOperator(shape, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y)
    sol, info = gmres(AxOp, RHS, rtol=tol, restart=500, x0=exact_sol, callback=lambda rk: print(f"GMRES residual: {np.linalg.norm(rk)}"))

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
    Lam_X_ExactList[0:Nib, k] = lam_x_exact
    Lam_Y_ExactList[0:Nib, k] = lam_y_exact

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
