import numpy as np
import matplotlib.pyplot as plt
from sksparse.cholmod import cholesky
from scipy.sparse import diags, kron, eye, csr_matrix, bmat, lil_matrix
from scipy.sparse.linalg import spsolve, gmres, splu, LinearOperator, minres
from scipy.interpolate import BSpline
from scipy.linalg import solve_triangular
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

def SchurLinearOperator_new(shape, lu_factorization, UGridX, UGridY, VGridX, VGridY,
                            xib, yib,
                            delta_x, delta_y,
                            N_U, N_V, N_P, Nib, cut):

    def mv(vec):
        Lam_x = vec[:Nib]
        Lam_y = vec[Nib:2*Nib]
        return apply_Schur_new(Lam_x, Lam_y, lu_factorization, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, cut)
        
    return LinearOperator((shape, shape), matvec=mv)

def SchurLinearOperator_mini(chol_U, chol_V, G_x, G_y, D_x, D_y, N_P):
    def matvec(p):
        # p: (Np,) vector
        # Step 1: compute G_x p and G_y p
        gx_p = G_x @ p
        gy_p = G_y @ p

        # Step 2: solve Lap_U * x = G_x p  and Lap_V * y = G_y p
        u = chol_U.solve_A(gx_p)
        v = chol_V.solve_A(gy_p)

        # Step 3: compute D_x u + D_y v
        return -(D_x @ u + D_y @ v)

    return LinearOperator((N_P, N_P), matvec=matvec, dtype=np.float64)

def schur_rhs_mini(rhs, 
                  chol_U, chol_V, D_x, D_y,
                  N_U, N_V, N_P):
    
    # Split incoming RHS vector
    f_u = rhs[:N_U]
    f_v = rhs[N_U:N_U + N_V]
    f_p = rhs[N_U + N_V:N_U + N_V + N_P]

    # Compute L_U^{-1} f_u and L_V^{-1} f_v efficiently
    u_tilde = chol_U.solve_A(f_u)
    v_tilde = chol_V.solve_A(f_v)

    # Combine into Schur RHS
    rhs_S = f_p - (D_x @ u_tilde + D_y @ v_tilde)

    return rhs_S

def apply_Ainv_Stokes(rhs, lu_factorization, N_U, N_V):

    x = lu_factorization.solve(rhs)

    u = x[:N_U]
    v = x[N_U:N_U+N_V]
    p = x[N_U+N_V:]
    
    return u, v, p

def apply_Schur_new(Lam_x, Lam_y, lu_factorization,
                    UGridX, UGridY, VGridX, VGridY,
                    xib, yib,
                    delta_x, delta_y,
                    N_U, N_V, N_P, cut):

    # B * [Lam_x, Lam_y]
    rhs_U = spreadQ_x(UGridX, UGridY, xib, yib, Lam_x, delta_x, cut)
    rhs_V = spreadQ_y(VGridX, VGridY, xib, yib, Lam_y, delta_y, cut)
    rhs_div = np.zeros(N_P) 

    # Solve A^{-1}
    U, V, _ = apply_Ainv_Stokes(np.concatenate([rhs_U, rhs_V, rhs_div]), lu_factorization, N_U, N_V)

    # C * (result)
    res_x = interpPhi_x(UGridX, UGridY, xib, yib, U, delta_x, cut)
    res_y = interpPhi_y(VGridX, VGridY, xib, yib, V, delta_y, cut)

    return np.concatenate([res_x, res_y])

def compute_U_V_P_postprocessing(Lam_x, Lam_y, lu_factorization,
                                    UGridX, UGridY, VGridX, VGridY,
                                    xib, yib,
                                    delta_x, delta_y,
                                    f_BC, g_BC, h_BC,
                                    N_U, N_V, cut):
    rhs_U = f_BC - spreadQ_x(UGridX, UGridY, xib, yib, Lam_x, delta_x, cut)
    rhs_V = g_BC - spreadQ_y(VGridX, VGridY, xib, yib, Lam_y, delta_y, cut)
    rhs_P = h_BC

    U, V, P = apply_Ainv_Stokes(np.concatenate([rhs_U, rhs_V, rhs_P]), lu_factorization, N_U, N_V)

    return U, V, P    

def compute_U_postprocessing(Pi, Lam_x, UGridX, UGridY, xib, yib, Lap_U, G_x, delta_x, f_BC, cut):
    RHS = f_BC + G_x @ Pi - spreadQ_x(UGridX, UGridY, xib, yib, Lam_x, delta_x, cut)
    U = spsolve(Lap_U, RHS)
    return U
    
def compute_V_postprocessing(Pi, Lam_y, VGridX, VGridY, xib, yib, Lap_V, G_y, delta_y, g_BC, cut):
    RHS = g_BC + G_y @ Pi - spreadQ_y(VGridX, VGridY, xib, yib, Lam_y, delta_y, cut)
    V = spsolve(Lap_V, RHS)
    return V

def schur_rhs_new(rhs, lu_factorization,
                  UGridX, UGridY, VGridX, VGridY,
                  delta_x, delta_y, xib, yib,
                  N_U, N_V, N_P, Nib, cut):

    # unpack
    rhs_U = rhs[:N_U]
    rhs_V = rhs[N_U:N_U+N_V]

    # the original rhs includes a pressure row and two IB rows, but
    # pressure is now solved inside A^{-1}, so we skip rhs_P
    offset = N_U + N_V
    rhs_div = rhs[offset:offset + N_P]
    offset += N_P
    rhs_Jx = rhs[offset:offset + Nib]
    rhs_Jy = rhs[offset + Nib:]

    # STEP 1: Apply A^{-1} to (rhs_U, rhs_V, rhs_div)
    U, V, _ = apply_Ainv_Stokes(np.concatenate([rhs_U, rhs_V, rhs_div]), lu_factorization, N_U, N_V)

    # STEP 2: Apply C to the result
    CAinv_x = interpPhi_x(UGridX, UGridY, xib, yib, U, delta_x, cut)
    CAinv_y = interpPhi_y(VGridX, VGridY, xib, yib, V, delta_y, cut)

    # STEP 3: Schur RHS = -C*A^{-1}(rhs)  +  original IB rhs components
    # (these rhs_Jx, rhs_Jy represent forcing on the constraints)
    schur_rhs_x = CAinv_x - rhs_Jx
    schur_rhs_y = CAinv_y - rhs_Jy

    return np.concatenate((schur_rhs_x, schur_rhs_y))

def build_staggered_Laps(Nx, dx):
    dx2 = dx**2

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

    Lap_U = kron(D2, eye(Nx + 1, format='csr'), format='csr') + kron(eye(Nx, format='csr'), D2_p, format='csr')
    Lap_V = kron(D2_p, eye(Nx, format='csr'), format='csr') + kron(eye(Nx + 1, format='csr'), D2, format='csr')

    return Lap_U, Lap_V

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
def solve_from_svd(U, Sigma, Vh, rhs):
    Sigma_inv_diag = 1 / Sigma

    Sigma_inv = np.diag(Sigma_inv_diag)

    res = Vh.conj().T @ Sigma_inv @ U.T @ rhs

    return res

# solve the system, assumes Dirichlet boundaries = 0 for x and y dimensions
def solve(left_bound, right_bound, lu_factorization, f_bc, g_bc, h_bc, z_x, z_y, rad, Nx, tol, cut):
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

    # # # preconditioner for stokes system
    # Lap_P = D_x @ G_x + D_y @ G_y
    # diag = Lap_P.diagonal()
    # diag_safe = np.where(np.abs(diag) > 0, diag, 1.0)
    # def apply_Minv(r): return r / diag_safe
    # P_op = LinearOperator((N_P, N_P), matvec=apply_Minv, dtype=np.float64)

    # M_U = diags(1.0 / Lap_U.diagonal())
    # M_V = diags(1.0 / Lap_V.diagonal())

    # preconditioner = D_x @ M_U @ G_x + D_y @ M_V @ G_y

    RHS = np.concatenate([f_bc, g_bc, h_bc, z_x, z_y])
    RHS_schur = schur_rhs_new(RHS, lu_factorization, UGridX, UGridY, VGridX, VGridY, delta_x, delta_y, xib, yib, N_U, N_V, N_P, Nib, cut)

    # Solve
    shape = 2 * Nib
    SchurOp = SchurLinearOperator_new(shape, lu_factorization, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, cut)
    sol, _ = gmres(SchurOp, RHS_schur, rtol=tol, restart=500, callback=lambda rk: print(f"GMRES residual: {np.linalg.norm(rk)}"))

    # # Solve using dense Schur matrix
    # shape = 2 * Nib
    # schurDense = np.zeros((shape, shape))
    # for col in range(shape): 
    #     eye_mat = np.zeros(shape)
    #     eye_mat[col] = 1
    #     eye_Lam_x = eye_mat[:Nib]
    #     eye_Lam_y = eye_mat[Nib:]
    #     res = apply_Schur_new(eye_Lam_x, eye_Lam_y, lu_factorization, UGridX, UGridY, VGridX, VGridY, xiv, yiv, delta_x, delta_y, N_U, N_V, N_P, cut)
    #     schurDense[:,col] = res

    # # Compute SVD once
    # U, Sigma, Vh = np.linalg.svd(schurDense)

    # sol = solve_from_svd(U, Sigma, Vh, RHS_schur)

    # Split (no change in ordering of partition)
    lam_X = sol[:Nib]
    lam_Y = sol[Nib:]

    # Postprocessing: compute U and V
    U, V, P = compute_U_V_P_postprocessing(lam_X, lam_Y, lu_factorization, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, f_bc, g_bc, h_bc, N_U, N_V, cut)

    # # COMPUTE USING FULL OPERATOR CHECK
    # Nu = apply_A(np.concatenate([U, V, P, lam_X, lam_Y]), UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y, cut)
    # residual_check_N = np.linalg.norm(Nu - RHS) / np.linalg.norm(RHS)
    # print(f'residual Nu = {residual_check_N}')

    return U, V, P, lam_X, lam_Y

def solve_from_svd(U, Sigma, Vh, rhs):
    Sigma_inv_diag = 1 / Sigma

    Sigma_inv = np.diag(Sigma_inv_diag)

    res = Vh.conj().T @ Sigma_inv @ U.T @ rhs

    return res

# solve the system using a factorized dense Schur complement, assumes Dirichlet boundaries = 0 for x and y dimensions
def solve_factorized(left_bound, right_bound, lu_factorization, U_schur, Sigma_schur, Vh_schur, f_bc, g_bc, h_bc, z_x, z_y, rad, Nx, tol, cut):
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

    # # # preconditioner for stokes system
    # Lap_P = D_x @ G_x + D_y @ G_y
    # diag = Lap_P.diagonal()
    # diag_safe = np.where(np.abs(diag) > 0, diag, 1.0)
    # def apply_Minv(r): return r / diag_safe
    # P_op = LinearOperator((N_P, N_P), matvec=apply_Minv, dtype=np.float64)

    # M_U = diags(1.0 / Lap_U.diagonal())
    # M_V = diags(1.0 / Lap_V.diagonal())

    # preconditioner = D_x @ M_U @ G_x + D_y @ M_V @ G_y

    RHS = np.concatenate([f_bc, g_bc, h_bc, z_x, z_y])
    RHS_schur = schur_rhs_new(RHS, lu_factorization, UGridX, UGridY, VGridX, VGridY, delta_x, delta_y, xib, yib, N_U, N_V, N_P, Nib, cut)

    # Solve
    sol = solve_from_svd(U_schur, Sigma_schur, Vh_schur, RHS_schur)
    
    # Split (no change in ordering of partition)
    lam_X = sol[:Nib]
    lam_Y = sol[Nib:]

    # Postprocessing: compute U and V
    U, V, P = compute_U_V_P_postprocessing(lam_X, lam_Y, lu_factorization, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, f_bc, g_bc, h_bc, N_U, N_V, cut)

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
