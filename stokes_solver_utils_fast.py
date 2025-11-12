import numpy as np
import matplotlib.pyplot as plt
from sksparse.cholmod import cholesky
from scipy.sparse import diags, kron, eye, csr_matrix, bmat, lil_matrix
from scipy.sparse.linalg import spsolve, gmres, splu, LinearOperator, minres
from scipy.interpolate import BSpline
from numba import jit

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

def interpPhi_x(Xu, Yu, xq, yq, U, delta_x, cut):
    """
    Interpolate u(x,y) from staggered x-face grid (Xu, Yu)
    to Lagrangian points (xq, yq) using composite δ_x with local support.
    """
    if U.ndim == 1:
        U = U.reshape(Xu.shape, order='F')

    Jphi = np.zeros_like(xq)
    dx = Xu[0,1] - Xu[0,0]
    dy = Yu[1,0] - Yu[0,0]
    Ny, Nx = Xu.shape

    for k in range(len(xq)):
        xk, yk = xq[k], yq[k]

        # Local index domain
        i_min = max(int((xk - cut - Xu[0,0]) / dx), 0)
        i_max = min(int((xk + cut - Xu[0,0]) / dx) + 1, Nx)
        j_min = max(int((yk - cut - Yu[0,0]) / dy), 0)
        j_max = min(int((yk + cut - Yu[0,0]) / dy) + 1, Ny)

        Xloc = Xu[j_min:j_max, i_min:i_max]
        Yloc = Yu[j_min:j_max, i_min:i_max]
        Uloc = U[j_min:j_max, i_min:i_max]

        # Same delta usage as original
        phi = delta_x(Xloc, Yloc, xk, yk)

        Jphi[k] = dx * dy * np.sum(Uloc * phi)

    return Jphi


def interpPhi_y(Xv, Yv, xq, yq, V, delta_y, cut):
    """
    Interpolate v(x,y) from staggered y-face grid (Xv, Yv)
    to Lagrangian points (xq, yq) using composite δ_y with local support.
    """
    if V.ndim == 1:
        V = V.reshape(Xv.shape, order='F')

    Jphi = np.zeros_like(xq)
    dx = Xv[0,1] - Xv[0,0]
    dy = Yv[1,0] - Yv[0,0]
    Ny, Nx = Xv.shape

    for k in range(len(xq)):
        xk, yk = xq[k], yq[k]

        i_min = max(int((xk - cut - Xv[0,0]) / dx), 0)
        i_max = min(int((xk + cut - Xv[0,0]) / dx) + 1, Nx)
        j_min = max(int((yk - cut - Yv[0,0]) / dy), 0)
        j_max = min(int((yk + cut - Yv[0,0]) / dy) + 1, Ny)

        Xloc = Xv[j_min:j_max, i_min:i_max]
        Yloc = Yv[j_min:j_max, i_min:i_max]
        Vloc = V[j_min:j_max, i_min:i_max]

        phi = delta_y(Xloc, Yloc, xk, yk)

        Jphi[k] = dx * dy * np.sum(Vloc * phi)

    return Jphi

def spreadQ_x(Xu, Yu, xq, yq, qx, delta_x, cut):
    """
    Spread Lagrangian x-forces qx to Eulerian x-face grid (Xu, Yu)
    using composite δ_x with local support (same delta call signature).
    """
    Fx = np.zeros_like(Xu)

    # approximate segment length
    xp = np.asarray(xq)
    yp = np.asarray(yq)
    dxs = np.sqrt(np.diff(np.concatenate([xp, xp[:1]]))**2 +
                  np.diff(np.concatenate([yp, yp[:1]]))**2)
    ds = np.mean(dxs)

    dx = Xu[0,1] - Xu[0,0]
    dy = Yu[1,0] - Yu[0,0]
    Ny, Nx = Xu.shape

    for k in range(len(qx)):
        xk, yk = xq[k], yq[k]

        i_min = max(int((xk - cut - Xu[0,0]) / dx), 0)
        i_max = min(int((xk + cut - Xu[0,0]) / dx) + 1, Nx)
        j_min = max(int((yk - cut - Yu[0,0]) / dy), 0)
        j_max = min(int((yk + cut - Yu[0,0]) / dy) + 1, Ny)

        Xloc = Xu[j_min:j_max, i_min:i_max]
        Yloc = Yu[j_min:j_max, i_min:i_max]

        phi = delta_x(Xloc, Yloc, xk, yk)

        Fx[j_min:j_max, i_min:i_max] += qx[k] * phi * ds

    return Fx.ravel(order='F')


def spreadQ_y(Xv, Yv, xq, yq, qy, delta_y, cut):
    """
    Spread Lagrangian y-forces qy to Eulerian y-face grid (Xv, Yv)
    using composite δ_y with local support (same delta call signature).
    """
    Fy = np.zeros_like(Xv)

    xp = np.asarray(xq)
    yp = np.asarray(yq)
    dxs = np.sqrt(np.diff(np.concatenate([xp, xp[:1]]))**2 +
                  np.diff(np.concatenate([yp, yp[:1]]))**2)
    ds = np.mean(dxs)

    dx = Xv[0,1] - Xv[0,0]
    dy = Yv[1,0] - Yv[0,0]
    Ny, Nx = Xv.shape

    for k in range(len(qy)):
        xk, yk = xq[k], yq[k]

        i_min = max(int((xk - cut - Xv[0,0]) / dx), 0)
        i_max = min(int((xk + cut - Xv[0,0]) / dx) + 1, Nx)
        j_min = max(int((yk - cut - Yv[0,0]) / dy), 0)
        j_max = min(int((yk + cut - Yv[0,0]) / dy) + 1, Ny)

        Xloc = Xv[j_min:j_max, i_min:i_max]
        Yloc = Yv[j_min:j_max, i_min:i_max]

        phi = delta_y(Xloc, Yloc, xk, yk)

        Fy[j_min:j_max, i_min:i_max] += qy[k] * phi * ds

    return Fy.ravel(order='F')


def AxLinearOperator(shape, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y):
    n = shape
    def mv(unknowns):
        return apply_A(unknowns, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y)
    return LinearOperator((n, n), matvec=mv)

def apply_A(x, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y, cut):
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
    res[0:N_U] = Lap_U @ U - G_x @ P + spreadQ_x(UGridX, UGridY, xib, yib, Lam_X, delta_x, cut)
    offset = N_U
    res[offset:offset + N_V] = Lap_V @ V - G_y @ P + spreadQ_y(VGridX, VGridY, xib, yib, Lam_Y, delta_y, cut)
    offset = offset + N_V
    res[offset:offset + N_P] = D_x @ U + D_y @ V
    offset = offset + N_P
    res[offset:offset + Nib] = interpPhi_x(UGridX, UGridY, xib, yib, U, delta_x, cut)
    offset = offset + Nib
    res[offset:offset + Nib] = interpPhi_y(VGridX, VGridY, xib, yib, V, delta_y, cut)
    
    return res

def SchurLinearOperator_new(shape, UGridX, UGridY, VGridX, VGridY,
                            xib, yib,
                            delta_x, delta_y,
                            Lap_U, Lap_V, LU_U, LU_V, G_x, G_y, D_x, D_y,
                            N_U, N_V, N_P, Nib, tol, cut):

    def mv(vec):
        Lam_x = vec[:Nib]
        Lam_y = vec[Nib:2*Nib]
        return apply_Schur_new(Lam_x, Lam_y,
                               UGridX, UGridY, VGridX, VGridY,
                               xib, yib, delta_x, delta_y,
                               Lap_U, Lap_V, LU_U, LU_V, G_x, G_y, D_x, D_y,
                               N_U, N_V, N_P, tol, cut)

    return LinearOperator((shape, shape), matvec=mv)

def apply_Ainv_Stokes(rhs, Lap_U, Lap_V, LU_U, LU_V, G_x, G_y, D_x, D_y,
                      N_U, N_V, N_P, tol):
    
    # Build full sparse L operator
    # Zero blocks for bmat (explicit shapes)
    Z_UV = csr_matrix((N_U, N_V))
    Z_VU = csr_matrix((N_V, N_U))
    Z_PP = csr_matrix((N_P, N_P))

    # Saddle point system
    L = bmat([
        [Lap_U, Z_VU,  -G_x],
        [Z_UV,  Lap_V, -G_y],
        [D_x,   D_y,   Z_PP] 
    ], format='csr')

    # Preconditioner apply: y -> M^{-1} y (block diagonal)
    def apply_minv(y):
        # y is a vector of length N_U+N_V+N_P
        yu = y[:N_U]
        yv = y[N_U:N_U + N_V]
        yp = y[N_U + N_V:]

        xu = LU_U.solve(yu)         # solve Lap_U * xu = yu
        xv = LU_V.solve(yv)         # solve Lap_V * xv = yv
        xp = yp.copy()              # keep pressure block

        return np.concatenate([xu, xv, xp])

    M_inv = LinearOperator((N_U + N_V + N_P, N_U + N_V + N_P), matvec=apply_minv, dtype=float)

    # Solve using MINRES
    x, info = minres(L, rhs, M=M_inv, rtol=tol)
    if info != 0:
        # info > 0 : maxiter reached; info < 0 : illegal input or breakdown
        print(f"MINRES returned info = {info} (non-zero). You may need a better preconditioner.")
    # split result
    x_u = x[:N_U]
    x_v = x[N_U:N_U + N_V]
    x_p = x[N_U + N_V:]

    return x_u, x_v, x_p

def apply_Schur_new(Lam_x, Lam_y,
                    UGridX, UGridY, VGridX, VGridY,
                    xib, yib,
                    delta_x, delta_y,
                    Lap_U, Lap_V, LU_U, LU_V, G_x, G_y, D_x, D_y,
                    N_U, N_V, N_P, tol, cut):

    # B * [Lam_x, Lam_y]
    rhs_U = spreadQ_x(UGridX, UGridY, xib, yib, Lam_x, delta_x, cut)
    rhs_V = spreadQ_y(VGridX, VGridY, xib, yib, Lam_y, delta_y, cut)
    rhs_div = np.zeros(N_P)  # No divergence forcing here

    # Solve A^{-1}
    U, V, _ = apply_Ainv_Stokes(np.concatenate([rhs_U, rhs_V, rhs_div]), Lap_U, Lap_V, LU_U, LU_V, G_x, G_y, D_x, D_y, N_U, N_V, N_P, tol)

    # C * (result)
    res_x = interpPhi_x(UGridX, UGridY, xib, yib, U, delta_x, cut)
    res_y = interpPhi_y(VGridX, VGridY, xib, yib, V, delta_y, cut)

    return np.concatenate([res_x, res_y])

def compute_U_postprocessing(Pi, Lam_x, UGridX, UGridY, xib, yib, Lap_U, G_x, delta_x, f_BC, cut):
    RHS = f_BC + G_x @ Pi - spreadQ_x(UGridX, UGridY, xib, yib, Lam_x, delta_x, cut)
    U = spsolve(Lap_U, RHS)
    return U
    
def compute_V_postprocessing(Pi, Lam_y, VGridX, VGridY, xib, yib, Lap_V, G_y, delta_y, g_BC, cut):
    RHS = g_BC + G_y @ Pi - spreadQ_y(VGridX, VGridY, xib, yib, Lam_y, delta_y, cut)
    V = spsolve(Lap_V, RHS)
    return V

def schur_rhs_new(rhs, 
                  Lap_U, Lap_V, LU_U, LU_V, G_x, G_y, D_x, D_y,
                  UGridX, UGridY, VGridX, VGridY,
                  delta_x, delta_y, xib, yib,
                  N_U, N_V, N_P, Nib, tol, cut):

    # unpack
    rhs_U = rhs[:N_U]
    rhs_V = rhs[N_U:N_U+N_V]

    # the original rhs includes a pressure row and two IB rows, but
    # pressure is now solved inside A^{-1}, so we skip rhs_P
    offset = N_U + N_V
    rhs_div = rhs[offset:offset + N_P]
    offset += N_P
    rhs_Jx = rhs[offset:offset + Nib]
    rhs_Jy = rhs[offset + Nib:offset + 2*Nib]

    # STEP 1: Apply A^{-1} to (rhs_U, rhs_V, rhs_div)
    U, V, _ = apply_Ainv_Stokes(np.concatenate([rhs_U, rhs_V, rhs_div]), Lap_U, Lap_V, LU_U, LU_V, G_x, G_y, D_x, D_y,
                                N_U, N_V, N_P, tol)

    # STEP 2: Apply C to the result
    CAinv_x = interpPhi_x(UGridX, UGridY, xib, yib, U, delta_x, cut)
    CAinv_y = interpPhi_y(VGridX, VGridY, xib, yib, V, delta_y, cut)

    # STEP 3: Schur RHS = -C*A^{-1}(rhs)  +  original IB rhs components
    # (these rhs_Jx, rhs_Jy represent forcing on the constraints)
    schur_rhs_x = rhs_Jx - CAinv_x
    schur_rhs_y = rhs_Jy - CAinv_y

    return np.concatenate((schur_rhs_x, schur_rhs_y))

def build_staggered_Laps(Nx, dx):
    """
    Construct Lap_U and Lap_V using D^T D on the staggered MAC grids,
    maintaining the same sizes and indexing order as your original code.
    """

    dx2 = dx**2

    # --- 1D second-derivative stencils (Dirichlet) ---
    # matches your original D2_p and D2 exactly (including -3 diagonal closure)
    e_p = np.ones(Nx + 1)
    e   = np.ones(Nx)

    D2_p = diags([e_p, -2*e_p, e_p], offsets=[-1, 0, 1],
                 shape=(Nx + 1, Nx + 1), format='lil')
    D2_p[0, 0] = -3.0
    D2_p[-1, -1] = -3.0
    D2_p = (D2_p / dx2).tocsr()

    D2 = diags([e, -2*e, e], offsets=[-1, 0, 1],
               shape=(Nx, Nx), format='csr') / dx2

    # --- 1D first-difference stencils (Dirichlet) ---
    # forward/back difference pair consistent with the BC closure
    D1_p = diags([e_p, -e_p], offsets=[0, -1], shape=(Nx + 1, Nx + 1), format='csr') / dx
    D1   = diags([e,   -e  ], offsets=[0, -1], shape=(Nx, Nx), format='csr') / dx

    # --- U-grid derivative operators ---
    # U is (Nx+1) x Nx → size N_U = (Nx+1)*Nx
    Dux = kron(D1_p, eye(Nx, format='csr'), format='csr')    # derivative in x-direction on U grid
    Duy = kron(eye(Nx + 1, format='csr'), D1, format='csr')  # derivative in y-direction on U grid

    Lap_U = (Dux.T @ Dux) + (Duy.T @ Duy)

    # --- V-grid derivative operators ---
    # V is Nx x (Nx+1) → size N_V = Nx*(Nx+1)
    Dvx = kron(D1, eye(Nx + 1, format='csr'), format='csr')  # derivative in x-direction on V grid
    Dvy = kron(eye(Nx, format='csr'), D1_p, format='csr')    # derivative in y-direction on V grid

    Lap_V = (Dvx.T @ Dvx) + (Dvy.T @ Dvy)

    return Lap_U.tocsr(), Lap_V.tocsr()

def build_staggered_Grads_Divs(Nx, dx):
    """
    Build staggered-grid divergence and its (negative) transpose gradient
    using the same shapes/ordering as your original implementation.

    Grid layout assumptions (matching your original code):
      - U lives on vertical faces:       N_U = Nx * (Nx + 1)
      - V lives on horizontal faces:     N_V = (Nx + 1) * Nx  (same number as N_U)
      - P lives on a (Nx+1) x (Nx+1) mesh: N_P = (Nx + 1)**2

    These sizes match the kron expansions in your original snippet.
    Dirichlet boundaries are implicitly handled by the 1D stencil shapes.
    """
    # 1D forward-like stencil used in original (shape (Nx+1, Nx))
    # (this matches your original Dx_f / Dy_b creation)
    stencil_rows = np.ones(Nx + 1)
    # shape (Nx+1, Nx): element i uses difference between col i and col i-1 (offsets 0 and -1)
    D1 = diags([stencil_rows, -stencil_rows], offsets=[0, -1], shape=(Nx + 1, Nx), format='csr') / dx

    # Note: original code used D_y = kron(I_{Nx+1}, D_y_backward)
    # and D_x = kron(D_x_forward, I_{Nx+1}).
    # Keep exactly those kron orders so shapes match your existing code.
    D_y = kron(eye(Nx + 1, format='csr'), D1, format='csr')   # maps V -> P
    D_x = kron(D1, eye(Nx + 1, format='csr'), format='csr')   # maps U -> P

    # Gradients are exact adjoints (negative transposes)
    G_x = (-D_x.transpose()).tocsr()
    G_y = (-D_y.transpose()).tocsr()

    return G_x, G_y, D_x, D_y

# @jit(nopython=True)
# def solve_from_svd(U, Sigma, Vh, rhs):
#     Sigma_inv_diag = 1 / Sigma

#     Sigma_inv = np.diag(Sigma_inv_diag)

#     res = Vh.conj().T @ Sigma_inv @ U.T @ rhs

#     return res

# solve the system, assumes Dirichlet boundaries = 0 for x and y dimensions
def solve(left_bound, right_bound, f_bc, g_bc, h_bc, z_x, z_y, rad, Nx, tol, cut):
    x = np.linspace(left_bound, right_bound, Nx + 2)
    dx = x[1] - x[0]
    y = x.copy()
    dy = dx

    # IB
    dth = dx / rad
    theta = np.arange(0, 2*np.pi - dth, dth)
    Nib = len(theta)
    xib = rad * np.cos(theta) 
    yib = rad * np.sin(theta)
    # FOR TESTING ONLY
    # xib = 0.5 + rad * np.cos(theta) 
    # yib = 0.5 + rad * np.sin(theta)

    delta_x, delta_y = make_composite_deltas(dx, n=3)

    x_trunc = x[1:-1]    # length Nx
    y_trunc = y[1:-1]    # length Ny
    x_mid = x + dx / 2
    y_mid = y + dy / 2
    x_offset = x_mid[:-1]
    y_offset = y_mid[:-1]

    UGridX, UGridY = np.meshgrid(x_trunc, y_offset)
    VGridX, VGridY = np.meshgrid(x_offset, y_trunc)

    N_U = (Nx + 1) * Nx
    N_V = Nx * (Nx + 1)
    N_P = (Nx + 1) * (Nx + 1)

    Lap_U, Lap_V = build_staggered_Laps(Nx, dx)
    G_x, G_y, D_x, D_y = build_staggered_Grads_Divs(Nx, dx)

    # Prefactor Lap_U and Lap_V with sparse LU (for preconditioner solves)
    LU_U = splu(Lap_U)   # may raise if Lap_U is singular — check BCs
    LU_V = splu(Lap_V)

    RHS = np.concatenate([f_bc, g_bc, h_bc, z_x, z_y])
    RHS_schur = schur_rhs_new(RHS, Lap_U, Lap_V, LU_U, LU_V, G_x, G_y, D_x, D_y, UGridX, UGridY, VGridX, VGridY, delta_x, delta_y, xib, yib, N_U, N_V, N_P, Nib, tol, cut)

    # Solve
    shape = 2 * Nib
    SchurOp = SchurLinearOperator_new(shape, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, Lap_U, Lap_V, LU_U, LU_V, G_x, G_y, D_x, D_y, N_U, N_V, N_P, Nib, tol, cut)
    sol, _ = gmres(SchurOp, RHS_schur, rtol=tol, restart=500, callback=lambda rk: print(f"GMRES residual: {np.linalg.norm(rk)}"))

    # # Solve using dense Schur matrix
    # shape = N_P + 2 * Nib
    # schurDense = np.zeros((shape, shape))
    # for col in range(shape): 
    #     eye_mat = np.zeros(shape)
    #     eye_mat[col] = 1
    #     eye_P = eye_mat[:N_P]
    #     eye_Lam_x = eye_mat[N_P:N_P + Nib]
    #     eye_Lam_y = eye_mat[N_P + Nib:]
    #     res = apply_Schur([eye_P, eye_Lam_x, eye_Lam_y], UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, Lap_U, Lap_V, G_x, G_y, D_x, D_y, cut)
    #     schurDense[:,col] = res

    # # Compute SVD once
    # U, Sigma, Vh = np.linalg.svd(schurDense)

    # sol = solve_from_svd(U, Sigma, Vh, RHS_schur)

    # Split (no change in ordering of partition)
    P = sol[:N_P]
    lam_X = sol[N_P:N_P + Nib]
    lam_Y = sol[N_P + Nib:]

    #P = P - np.mean(P)

    # Postprocessing: compute U and V
    U = compute_U_postprocessing(P, lam_X, UGridX, UGridY, xib, yib, Lap_U, G_x, delta_x, f_bc, cut)
    V = compute_V_postprocessing(P, lam_Y, VGridX, VGridY, xib, yib, Lap_V, G_y, delta_y, g_bc, cut)

    # # COMPUTE USING FULL OPERATOR CHECK
    # Nu = apply_A(np.concatenate([U, V, P, lam_X, lam_Y]), UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y, cut)
    # residual_check_N = np.linalg.norm(Nu - RHS) / np.linalg.norm(RHS)
    # print(f'residual Nu = {residual_check_N}')

    return U, V, P, lam_X, lam_Y

#####################
######  test  #######
#####################

# L = 1.0
# rad = 0.25
# size = 5

# size_range = range(size)
# N = np.zeros(size)
# Nib_array = np.zeros(size)
# for s in size_range:
#     N[s] = 15*(size_range[s] + 3)

# # compute Nib_max
# Nx_max = int(N[-1])
# x_max = np.linspace(0, L, Nx_max + 2)
# dx_max = x_max[1] - x_max[0]
# dth_max = dx_max / rad
# theta_max = np.arange(0, 2*np.pi - dth_max, dth_max)
# Nib_max = len(theta_max)

# UNumericalList = np.zeros((int((N[-1] + 1) * N[-1]), size))
# UExactList = np.zeros((int((N[-1] + 1) * N[-1]), size))
# PNumericalList = np.zeros((int((N[-1] + 1) * (N[-1] + 1)), size))
# PExactList = np.zeros((int((N[-1] + 1) * (N[-1] + 1)), size))
# VNumericalList = np.zeros((int((N[-1] + 1) * N[-1]), size))
# VExactList = np.zeros((int((N[-1] + 1) * N[-1]), size))
# Lam_X_NumericalList = np.zeros((int(Nib_max), size))
# Lam_X_ExactList = np.ones((int(Nib_max), size))
# Lam_Y_NumericalList = np.zeros((int(Nib_max), size))
# Lam_Y_ExactList = np.ones((int(Nib_max), size))

# err2Norm_Lap_U_arr = np.zeros(size)
# err2Norm_Lap_V_arr = np.zeros(size)
# err2Norm_Dx_arr = np.zeros(size)
# err2Norm_Dy_arr = np.zeros(size)

# for k in size_range:
#     # Parameters
#     Nx = int(N[k])
#     Ny = Nx

#     U, V, P, lam_X, lam_Y, UExact, VExact, PExact, err2Norm_Lap_U, err2Norm_Lap_V, err2Norm_Dx_U, err2Norm_Dx_V, Nib = solve(0, L, rad, Nx)

#     # Append to solution array 
#     UNumericalList[0:(Nx + 1) * Ny, k] = U
#     VNumericalList[0:Nx * (Ny + 1), k] = V
#     PNumericalList[0:(Nx + 1) * (Nx + 1), k] = P
#     Lam_X_NumericalList[0:Nib, k] = lam_X
#     Lam_Y_NumericalList[0:Nib, k] = lam_Y
#     UExactList[0:(Nx + 1) * Ny, k] = UExact.ravel(order='F')
#     VExactList[0:Nx * (Ny + 1), k] = VExact.ravel(order='F')
#     PExactList[0:(Nx + 1) * (Nx + 1), k] = PExact.ravel(order='F')

#     # Reshape back using Fortran order to match MATLAB layout
#     Uplot = U.reshape((Ny + 1, Nx), order='F')
#     Vplot = V.reshape((Ny, Nx + 1), order='F')
#     Pplot = P.reshape((Ny + 1, Nx + 1), order='F')

#     # Build full arrays with ghost rows/cols similar to MATLAB, but Dirichlet zero ghosts
#     UFull = np.zeros((Ny + 2, Nx + 2))
#     UFull[1:Ny + 2, 1:Nx + 1] = Uplot

#     VFull = np.zeros((Ny + 2, Nx + 2))
#     VFull[1:Ny + 1, 1:Nx + 2] = Vplot

# # Check errors
# err2Norm_U = np.zeros(size)
# err2Norm_V = np.zeros(size)
# err2Norm_P = np.zeros(size)
# err2Norm_Lam_X = np.zeros(size)
# err2Norm_Lam_Y = np.zeros(size)
# for i in size_range:
#     err2Norm_U[i] = np.linalg.norm(UExactList[0:int((N[i] + 1) * N[i]), i] - UNumericalList[0:int((N[i] + 1) * N[i]), i], ord=2) / np.linalg.norm(UExactList[0:int((N[i] + 1) * N[i]),i], ord=2)
#     err2Norm_V[i] = np.linalg.norm(VExactList[0:int((N[i] + 1) * N[i]), i] - VNumericalList[0:int((N[i] + 1) * N[i]), i], ord=2) / np.linalg.norm(VExactList[0:int((N[i] + 1) * N[i]),i], ord=2)
#     err2Norm_P[i] = np.linalg.norm(PExactList[0:int((N[i] + 1) * (N[i] + 1)), i] - PNumericalList[0:int((N[i] + 1) * (N[i] + 1)), i], ord=2) / np.linalg.norm(PExactList[0:int((N[i] + 1) * (N[i] + 1)),i], ord=2)
#     err2Norm_Lam_X[i] = np.linalg.norm(Lam_X_ExactList[0:int(Nib_array[i]), i] - Lam_X_NumericalList[0:int(Nib_array[i]), i], ord=2) / np.linalg.norm(Lam_X_ExactList[0:int(Nib_array[i]), i], ord=2)
#     err2Norm_Lam_Y[i] = np.linalg.norm(Lam_Y_ExactList[0:int(Nib_array[i]), i] - Lam_Y_NumericalList[0:int(Nib_array[i]), i], ord=2) / np.linalg.norm(Lam_Y_ExactList[0:int(Nib_array[i]), i], ord=2) 

# h = 1.0 / N
# def rates(err):
#     r = []
#     for i in range(1, len(err)):
#         r.append(np.log(err[i]/err[i-1]) / np.log(h[i]/h[i-1]))
#     return np.array(r)

# print("U rates:", rates(err2Norm_U))
# print("V rates:", rates(err2Norm_V))
# print("P rates:", rates(err2Norm_P))
# print("Lam_X rates:", rates(err2Norm_Lam_X))
# print("Lam_Y rates:", rates(err2Norm_Lam_Y))

# plt.figure(figsize=(6,5))
# plt.loglog(N, err2Norm_Lap_U_arr, '--o', label=r'$L_U$')
# plt.loglog(N, err2Norm_Lap_V_arr, '--o', label=r'$L_V$')
# plt.loglog(N, err2Norm_Dx_arr, '--o', label=r'$D_x$')
# plt.loglog(N, err2Norm_Dy_arr, '--o', label=r'$D_y$')
# plt.loglog(N, h**2, '--', label='2nd order')
# plt.xlabel(r'$N$', fontsize=22)
# plt.ylabel(r'$Err$', fontsize=22)
# plt.title(r'Relative error of $u_n(x, y)$, $p=2$ norm', fontsize=22)
# plt.legend(fontsize=22, loc='upper right')  # matplotlib uses "upper right"

# plt.grid(True, which="both", ls="--", lw=0.5)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(6,5))
# plt.loglog(N, err2Norm_U, '--o', label=r'$Err_U$')
# plt.loglog(N, err2Norm_V, '--o', label=r'$Err_V$')
# plt.loglog(N, err2Norm_P, '--o', label=r'$Err_P$')
# plt.loglog(N, err2Norm_Lam_X, '--o', label=r'$Err_{\lambda_X}$')
# plt.loglog(N, err2Norm_Lam_Y, '--o', label=r'$Err_{\lambda_Y}$')
# plt.loglog(N, h**2, '--', label='2nd order')

# plt.xlabel(r'$N$', fontsize=22)
# plt.ylabel(r'$Err$', fontsize=22)
# plt.title(r'Relative error of $u_n(x, y)$, $p=2$ norm', fontsize=22)
# plt.legend(fontsize=22, loc='upper right')  # matplotlib uses "upper right"

# plt.grid(True, which="both", ls="--", lw=0.5)
# plt.tight_layout()
# plt.show()

# xplot = np.linspace(0, L, Nx + 2)
# yplot = np.linspace(0, L, Ny + 2)
# Xplot, Yplot = np.meshgrid(xplot, yplot)

# # Plotting (3D surface)
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
# cmap = plt.cm.spring
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(Xplot, Yplot, UFull, cmap=cmap, edgecolor='none')
# ax.set_title("U")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(Xplot, Yplot, VFull, cmap=cmap, edgecolor='none')
# ax.set_title("V")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(Xplot, Yplot, np.sqrt(UFull**2 + VFull**2), cmap=cmap, edgecolor='none')
# ax.set_title("|velocity|")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# plt.show()
