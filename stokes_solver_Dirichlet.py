import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, csr_matrix, bmat, lil_matrix
from scipy.sparse.linalg import spsolve, gmres, LinearOperator
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

def SchurLinearOperator(shape, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, Lap_U, Lap_V, G_x, G_y, D_x, D_y, N_P, Nib):
    n = shape
    def mv(unknowns):
        unknown_P = unknowns[:N_P]
        unknown_Lam_x = unknowns[N_P:N_P + Nib]
        unknown_Lam_y = unknowns[N_P + Nib:N_P + 2*Nib]
        return apply_Schur([unknown_P, unknown_Lam_x, unknown_Lam_y], UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, Lap_U, Lap_V, G_x, G_y, D_x, D_y)
    return LinearOperator((n, n), matvec=mv)

def apply_Ainv(Lap_U, Lap_V, target_vec):
    target_vec_1, target_vec_2 = target_vec

    # second and third blocks are straightforward
    result_vec_1 = spsolve(Lap_U, target_vec_1)
    result_vec_2 = spsolve(Lap_V, target_vec_2)

    return [result_vec_1, result_vec_2]

def apply_Schur(unknown_blocks, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, Lap_U, Lap_V, G_x, G_y, D_x, D_y):
    Pi, Lam_x, Lam_y = unknown_blocks

    # apply B 
    B_U = spreadQ_x(UGridX, UGridY, xib, yib, Lam_x, delta_x) - G_x @ Pi
    B_V = spreadQ_y(VGridX, VGridY, xib, yib, Lam_y, delta_y) - G_y @ Pi

    # apply A inverse 
    Ainv_B_U, Ainv_B_V = apply_Ainv(Lap_U, Lap_V, [B_U, B_V])

    # apply C 
    res_1 = D_x @ Ainv_B_U + D_y @ Ainv_B_V
    res_2 = interpPhi_x(UGridX, UGridY, xib, yib, Ainv_B_U, delta_x)
    res_3 = interpPhi_y(VGridX, VGridY, xib, yib, Ainv_B_V, delta_y)

    return np.concatenate([res_1, res_2, res_3])

def compute_U_postprocessing(Pi, Lam_x, UGridX, UGridY, xib, yib, Lap_U, G_x, delta_x, f_BC):
    RHS = f_BC + G_x @ Pi - spreadQ_x(UGridX, UGridY, xib, yib, Lam_x, delta_x)
    U = spsolve(Lap_U, RHS)
    return U
    
def compute_V_postprocessing(Pi, Lam_y, VGridX, VGridY, xib, yib, Lap_V, G_y, delta_y, g_BC):
    RHS = g_BC + G_y @ Pi - spreadQ_y(VGridX, VGridY, xib, yib, Lam_y, delta_y)
    V = spsolve(Lap_V, RHS)
    return V

def schur_rhs(rhs, Lap_U, Lap_V, D_x, D_y, delta_x, delta_y, UGridX, UGridY, VGridX, VGridY, xib, yib, N_U, N_V, N_P, Nib):
    rhs_1 = rhs[0:N_U]
    offset = N_U
    rhs_2 = rhs[offset:offset + N_V]
    offset = offset + N_V
    rhs_3 = rhs[offset:offset + N_P]
    offset = offset + N_P
    rhs_4 = rhs[offset:offset + Nib]
    offset = offset + Nib
    rhs_5 = rhs[offset:offset + Nib]

    Ainv_U, Ainv_V = apply_Ainv(Lap_U, Lap_V, [rhs_1, rhs_2])
    CAinv_1 = D_x @ Ainv_U + D_y @ Ainv_V
    CAinv_2 = interpPhi_x(UGridX, UGridY, xib, yib, Ainv_U, delta_x)
    CAinv_3 = interpPhi_y(VGridX, VGridY, xib, yib, Ainv_V, delta_y)

    schur_rhs_1 = CAinv_1 - rhs_3
    schur_rhs_2 = CAinv_2 - rhs_4
    schur_rhs_3 = CAinv_3 - rhs_5

    return np.concatenate((schur_rhs_1, schur_rhs_2, schur_rhs_3))

L = 1.0
rad = 0.25
size = 3

size_range = range(size)
N = np.zeros(size)
Nib_array = np.zeros(size)
for s in size_range:
    N[s] = 15*(size_range[s] + 3)
    #N[s] = 3 * (size_range[s] + 1)

# compute Nib_max
Nx_max = int(N[-1])
x_max = np.linspace(0, L, Nx_max + 2)
dx_max = x_max[1] - x_max[0]
dth_max = dx_max / rad
theta_max = np.arange(0, 2*np.pi - dth_max, dth_max)
Nib_max = len(theta_max)

# -----------------------------
# Manufactured solution (Dirichlet zero on all boundaries)
# -----------------------------
def compute_U(xx, yy):
    # zero on boundaries because sin(pi*0)=sin(pi*1)=0
    return np.sin(np.pi * xx) * np.sin(np.pi * yy)

def compute_V(xx, yy):
    # same form (could be different), also zero on boundaries
    return np.sin(np.pi * xx) * np.sin(np.pi * yy)

def compute_P(xx, yy):
    # pressure chosen similarly (arbitrary)
    return np.sin(np.pi * xx) * np.sin(np.pi * yy)

# For PDE: Delta(u) - d/dx P = f, Delta(v) - d/dy P = g
def compute_f(xx, yy):
    # Laplacian of sin(pi x) sin(pi y) = -2*pi^2 * sin(pi x) sin(pi y)
    # d/dx P = pi * cos(pi x) * sin(pi y)
    return -2 * (np.pi**2) * np.sin(np.pi * xx) * np.sin(np.pi * yy) - (np.pi * np.cos(np.pi * xx) * np.sin(np.pi * yy))

def compute_g(xx, yy):
    # d/dy P = pi * sin(pi x) * cos(pi y)
    return -2 * (np.pi**2) * np.sin(np.pi * xx) * np.sin(np.pi * yy) - (np.pi * np.sin(np.pi * xx) * np.cos(np.pi * yy))

def compute_Lap_U(xx, yy):
    return -2 * (np.pi**2) * np.sin(np.pi * xx) * np.sin(np.pi * yy)

def compute_Lap_V(xx, yy):
    return -2 * (np.pi**2) * np.sin(np.pi * xx) * np.sin(np.pi * yy)

def compute_Dx_U(xx, yy):
    return np.pi * np.cos(np.pi * xx) * np.sin(np.pi * yy)

def compute_Dy_V(xx, yy):
    return np.pi * np.sin(np.pi * xx) * np.cos(np.pi * yy)


UNumericalList = np.zeros((int((N[-1] + 1) * N[-1]), size))
UExactList = np.zeros((int((N[-1] + 1) * N[-1]), size))
PNumericalList = np.zeros((int((N[-1] + 1) * (N[-1] + 1)), size))
PExactList = np.zeros((int((N[-1] + 1) * (N[-1] + 1)), size))
VNumericalList = np.zeros((int((N[-1] + 1) * N[-1]), size))
VExactList = np.zeros((int((N[-1] + 1) * N[-1]), size))
Lam_X_NumericalList = np.zeros((int(Nib_max), size))
Lam_X_ExactList = np.zeros((int(Nib_max), size))
Lam_Y_NumericalList = np.zeros((int(Nib_max), size))
Lam_Y_ExactList = np.zeros((int(Nib_max), size))

err2Norm_Lap_U_arr = np.zeros(size)
err2Norm_Lap_V_arr = np.zeros(size)
err2Norm_Dx_arr = np.zeros(size)
err2Norm_Dy_arr = np.zeros(size)

for k in size_range:
    # Parameters
    Nx = int(N[k])
    Ny = Nx
    x = np.linspace(0, L, Nx + 2)
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

    x_trunc = x[1:-1]    # length Nx
    y_trunc = y[1:-1]    # length Ny
    x_mid = x + dx / 2
    y_mid = y + dy / 2
    x_offset = x_mid[:-1]
    y_offset = y_mid[:-1]

    UGridX, UGridY = np.meshgrid(x_trunc, y_offset)
    VGridX, VGridY = np.meshgrid(x_offset, y_trunc)

    N_U = (Nx + 1) * Ny
    N_V = Nx * (Ny + 1)
    N_P = (Nx + 1) * (Ny + 1)

    # RHS
    f = np.zeros((Ny + 1, Nx))
    g = np.zeros((Ny, Nx + 1))
    h = np.zeros((Ny + 1, Nx + 1))
    z_x = np.zeros(Nib)
    z_y = np.zeros(Nib)

    # Analytic solutions
    UExact = np.zeros((Ny + 1, Nx))
    VExact = np.zeros((Ny, Nx + 1))
    PExact = np.zeros((Ny + 1, Nx + 1))
    lam_x_exact = np.zeros(Nib)
    lam_y_exact = np.zeros(Nib)

    for j in range(Ny + 1):
        for i in range(Nx):
            f[j, i] = compute_f(x_trunc[i], y_offset[j])
            UExact[j, i] = compute_U(x_trunc[i], y_offset[j])

    for j in range(Ny + 1):
        for i in range(Nx + 1):
            PExact[j, i] = compute_P(x_offset[i], y_offset[j])

    for j in range(Ny):
        for i in range(Nx + 1):
            g[j, i] = compute_g(x_offset[i], y_trunc[j])
            VExact[j, i] = compute_V(x_offset[i], y_trunc[j])

    exact_sol = np.concatenate([PExact.ravel(order='F'), lam_x_exact, lam_y_exact])

    z_x[:] = interpPhi_x(UGridX, UGridY, xib, yib, UExact, delta_x)
    z_y[:] = interpPhi_y(VGridX, VGridY, xib, yib, VExact, delta_y)

    # Dirichlet zero on all boundaries
    V_lower = 0.0
    V_upper = 0.0

    f_bc_mat = np.zeros((Ny + 1, Nx))
    g_bc_mat = np.zeros((Ny, Nx + 1))
    h_bc_mat = np.zeros((Ny + 1, Nx + 1))

    # Spread lamdas are included as before.
    f_bc = f.ravel(order='F') + f_bc_mat.ravel(order='F') #spreadQ_x(UGridX, UGridY, xib, yib, lam_x_exact, delta_x) + f_bc_mat.ravel(order='F')

    # For V: with Dirichlet 0 on top/bottom, the previous additional terms that set ghost based values vanish:
    # g boundary correction stays zero
    g_bc_mat[0, :] = 0.0
    g_bc_mat[-1, :] = 0.0
    g_bc = g.ravel(order='F') + g_bc_mat.ravel(order='F') #spreadQ_y(VGridX, VGridY, xib, yib, lam_y_exact, delta_y) + g_bc_mat.ravel(order='F')

    # h_bc (ghost flux-like vector) zero for Dirichlet
    h_bc_mat[0, :] = 0.0
    h_bc_mat[-1, :] = 0.0
    h_bc = h_bc_mat.ravel(order='F')

    # 1-D operators (Dirichlet in x, Dirichlet in y via zero ghosts)
    # assumes Nx = Ny
    e_p = np.ones(Nx + 1)
    e = np.ones(Nx)
    D2_p = diags([e_p, -2*e_p, e_p], offsets=[-1, 0, 1], shape=(Nx + 1, Nx + 1), format='lil')
    D2_p[0, 0] = -3.0
    D2_p[-1, -1] = -3.0
    D2_p = (D2_p / dx2).tocsr()

    D2 = diags([e, -2*e, e], offsets=[-1, 0, 1], shape=(Nx, Nx), format='lil')
    D2 = (D2 / dx2).tocsr()

    # Laplacians (Kronecker products)
    Lap_U = kron(D2, eye(Ny + 1, format='csr'), format='csr') + kron(eye(Nx, format='csr'), D2_p, format='csr')
    Lap_V = kron(D2_p, eye(Ny, format='csr'), format='csr') + kron(eye(Nx + 1, format='csr'), D2, format='csr')

    # check Laplacians
    computed_Lap_U = Lap_U @ UExact.ravel(order='F')
    computed_Lap_V = Lap_V @ VExact.ravel(order='F')

    exact_Lap_U = np.zeros([Ny + 1, Nx])
    exact_Lap_V = np.zeros([Ny, Nx + 1])
    # compare with actual laplacian
    for j in range(Ny + 1):
        for i in range(Nx):
            exact_Lap_U[j, i] = compute_Lap_U(x_trunc[i], y_offset[j])
    for j in range(Ny):
        for i in range(Nx + 1):
            exact_Lap_V[j, i] = compute_Lap_V(x_offset[i], y_trunc[j])

    err2Norm_Lap_U = np.linalg.norm(exact_Lap_U.ravel(order='F') - computed_Lap_U, ord=2) / np.linalg.norm(exact_Lap_U.ravel(order='F'), ord=2)
    err2Norm_Lap_V = np.linalg.norm(exact_Lap_V.ravel(order='F') - computed_Lap_V, ord=2) / np.linalg.norm(exact_Lap_V.ravel(order='F'), ord=2)

    err2Norm_Lap_U_arr[k] = err2Norm_Lap_U
    err2Norm_Lap_V_arr[k] = err2Norm_Lap_V

    # # --- 1D backward-difference stencil ---
    # D_1d = diags([-1 * np.ones(Nx), np.ones(Nx)], [0, -1], shape=(Nx + 1, Nx), format='lil') / dx

    # # --- 2D Kronecker structure ---
    # # Fortran ordering: x varies fastest
    # Ix = eye(Nx + 1, format='csr')
    # Iy = eye(Ny + 1, format='csr')

    # D_x = kron(Ix, D_1d.tocsr(), format='csr')
    # D_y = kron(D_1d.tocsr(), Iy, format='csr')

    # Dx_1d = diags([-1*np.ones(Nx+1), np.ones(Nx+1)], [0, -1], shape=(Nx+1, Nx)) / dx
    # # shape: (Nx+1, Nx) → maps U_x (Nx) to pressure (Nx+1)

    # # kron with identity along y (Ny+1 rows)
    # D_y = kron(eye(Ny+1, format='csr'), Dx_1d, format='csr')

    # Dy_1d = diags([np.ones(Ny+1), -1*np.ones(Ny+1)], [1, 0], shape=(Ny+1, Ny)) / dy
    # # shape: (Ny+1, Ny) → maps V_y (Ny) to pressure (Ny+1)

    # # kron with identity along x (Nx+1 columns)
    # D_x = kron(Dy_1d, eye(Nx+1, format='csr'), format='csr')  # shape (N_P, N_V)

    # G_x = -D_x.transpose().tocsr()
    # G_y = -D_y.transpose().tocsr()

    def build_Dx_1d(Nx, dx):
        D = lil_matrix((Nx+1, Nx), dtype=float)
        # interior rows 1..Nx-1
        for i in range(1, Nx):
            D[i, i] = 1.0/dx
            D[i, i-1] = -1.0/dx
        # row 0 and row Nx: choose ghost/BC consistent values (example below)
        D[0, 0] = 1.0/dx         # placeholder
        D[Nx, Nx-1] = -1.0/dx    # placeholder
        return D.tocsr()

    def build_Dy_1d(Ny, dy):
        D = lil_matrix((Ny+1, Ny), dtype=float)
        for j in range(1, Ny):
            D[j, j] = 1.0/dy
            D[j, j-1] = -1.0/dy
        D[0, 0] = 1.0/dy
        D[Ny, Ny-1] = -1.0/dy
        return D.tocsr()

    Dy_1d = build_Dx_1d(Nx, dx)
    D_y = kron(eye(Nx+1, format='csr'), Dy_1d, format='csr')
    Dx_1d = build_Dy_1d(Ny, dy)
    D_x = kron(Dx_1d, eye(Ny+1, format='csr'), format='csr')
    G_x = -D_x.transpose().tocsr()
    G_y = -D_y.transpose().tocsr()

    # check Divergence operators
    computed_Dx_U = D_x @ UExact.ravel(order='F')
    computed_Dy_V = D_y @ VExact.ravel(order='F')

    exact_Dx_U = np.zeros([Ny + 1, Nx + 1])
    exact_Dy_V = np.zeros([Ny + 1, Nx + 1])
    # compare with actual laplacian
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            exact_Dx_U[j, i] = compute_Dx_U(x_offset[i], y_offset[j])
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            exact_Dy_V[j, i] = compute_Dy_V(x_offset[i], y_offset[j])

    err2Norm_Dx_U = np.linalg.norm(exact_Dx_U.ravel(order='F') - computed_Dx_U, ord=2) / np.linalg.norm(exact_Dx_U.ravel(order='F'), ord=2)
    err2Norm_Dy_V = np.linalg.norm(exact_Dy_V.ravel(order='F') - computed_Dy_V, ord=2) / np.linalg.norm(exact_Dy_V.ravel(order='F'), ord=2)

    err2Norm_Dx_arr[k] = err2Norm_Dx_U
    err2Norm_Dy_arr[k] = err2Norm_Dy_V

    RHS = np.concatenate([f_bc, g_bc, h_bc, z_x, z_y])
    RHS_schur = schur_rhs(RHS, Lap_U, Lap_V, D_x, D_y, delta_x, delta_y, UGridX, UGridY, VGridX, VGridY, xib, yib, N_U, N_V, N_P, Nib)

    # Solve
    shape = N_P + 2 * Nib
    SchurOp = SchurLinearOperator(shape, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, Lap_U, Lap_V, G_x, G_y, D_x, D_y, N_P, Nib)
    sol, info = gmres(SchurOp, RHS_schur, rtol=tol, restart=500, x0=exact_sol, callback=lambda rk: print(f"GMRES residual: {np.linalg.norm(rk)}"))

    # Split (no change in ordering of partition)
    P = sol[:N_P]
    lam_X = sol[N_P:N_P + Nib]
    lam_Y = sol[N_P + Nib:]

    #P = P - np.mean(P)

    # Postprocessing: compute U and V
    U = compute_U_postprocessing(P, lam_X, UGridX, UGridY, xib, yib, Lap_U, G_x, delta_x, f_bc)
    V = compute_V_postprocessing(P, lam_Y, VGridX, VGridY, xib, yib, Lap_V, G_y, delta_y, g_bc)

    # Append to solution array 
    UNumericalList[0:(Nx + 1) * Ny, k] = U
    VNumericalList[0:Nx * (Ny + 1), k] = V
    PNumericalList[0:(Nx + 1) * (Nx + 1), k] = P
    Lam_X_NumericalList[0:Nib, k] = lam_X
    Lam_Y_NumericalList[0:Nib, k] = lam_Y
    UExactList[0:(Nx + 1) * Ny, k] = UExact.ravel(order='F')
    VExactList[0:Nx * (Ny + 1), k] = VExact.ravel(order='F')
    PExactList[0:(Nx + 1) * (Nx + 1), k] = PExact.ravel(order='F')
    Lam_X_ExactList[0:Nib, k] = lam_x_exact
    Lam_Y_ExactList[0:Nib, k] = lam_y_exact

    # Reshape back using Fortran order to match MATLAB layout
    Uplot = U.reshape((Ny + 1, Nx), order='F')
    Vplot = V.reshape((Ny, Nx + 1), order='F')
    Pplot = P.reshape((Ny + 1, Nx + 1), order='F')

    # Build full arrays with ghost rows/cols similar to MATLAB, but Dirichlet zero ghosts
    UFull = np.zeros((Ny + 2, Nx + 2))
    UFull[1:Ny + 2, 1:Nx + 1] = Uplot

    VFull = np.zeros((Ny + 2, Nx + 2))
    VFull[1:Ny + 1, 1:Nx + 2] = Vplot

# Check errors
err2Norm_U = np.zeros(size)
err2Norm_V = np.zeros(size)
err2Norm_P = np.zeros(size)
err2Norm_Lam_X = np.zeros(size)
err2Norm_Lam_Y = np.zeros(size)
for i in size_range:
    err2Norm_U[i] = np.linalg.norm(UExactList[0:int((N[i] + 1) * N[i]), i] - UNumericalList[0:int((N[i] + 1) * N[i]), i], ord=2) / np.linalg.norm(UExactList[0:int((N[i] + 1) * N[i]),i], ord=2)
    err2Norm_V[i] = np.linalg.norm(VExactList[0:int((N[i] + 1) * N[i]), i] - VNumericalList[0:int((N[i] + 1) * N[i]), i], ord=2) / np.linalg.norm(VExactList[0:int((N[i] + 1) * N[i]),i], ord=2)
    err2Norm_P[i] = np.linalg.norm(PExactList[0:int((N[i] + 1) * (N[i] + 1)), i] - PNumericalList[0:int((N[i] + 1) * (N[i] + 1)), i], ord=2) / np.linalg.norm(PExactList[0:int((N[i] + 1) * (N[i] + 1)),i], ord=2)
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
plt.loglog(N, err2Norm_Lap_U_arr, '--o', label=r'$L_U$')
plt.loglog(N, err2Norm_Lap_V_arr, '--o', label=r'$L_V$')
plt.loglog(N, err2Norm_Dx_arr, '--o', label=r'$D_x$')
plt.loglog(N, err2Norm_Dy_arr, '--o', label=r'$D_y$')
plt.loglog(N, h**2, '--', label='2nd order')
plt.xlabel(r'$N$', fontsize=22)
plt.ylabel(r'$Err$', fontsize=22)
plt.title(r'Relative error of $u_n(x, y)$, $p=2$ norm', fontsize=22)
plt.legend(fontsize=22, loc='upper right')  # matplotlib uses "upper right"

plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.show()

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
