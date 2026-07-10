import numpy as np
import matplotlib.pyplot as plt
from sksparse.cholmod import cholesky
from scipy.sparse import diags, kron, eye, csr_matrix, bmat, lil_matrix
from scipy.sparse.linalg import spsolve, gmres, splu, LinearOperator, minres
from scipy.interpolate import BSpline
from scipy.linalg import solve_triangular
from numba import jit

def build_K(xib, yib, offset, Nib): 
    offset_x, offset_y = offset
    K = np.zeros([2 * Nib, 3])
    for i in range(Nib): 
        rx = xib[i] + offset_x
        ry = yib[i] + offset_y
        K[2 * i, :]     = np.array([1, 0, -ry])
        K[2 * i + 1, :] = np.array([0, 1,  rx])
    return K
    

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

def apply_A_K(x, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y, K, cut):    
    U = x[0:N_U]
    offset = N_U
    V = x[offset:offset + N_V]
    offset = offset + N_V
    P = x[offset:offset + N_P]
    offset = offset + N_P
    Lam = x[offset:offset + 2 * Nib]
    offset = offset + 2 * Nib
    V_rigid = x[offset:]

    # deinterleave
    Lam_X = Lam[0::2]
    Lam_Y = Lam[1::2]

    K_res = K @ V_rigid

    res = np.zeros_like(x)
    res[0:N_U] = Lap_U @ U - G_x @ P + spreadQ_x(UGridX, UGridY, xib, yib, Lam_X, delta_x, cut)
    offset = N_U
    res[offset:offset + N_V] = Lap_V @ V - G_y @ P + spreadQ_y(VGridX, VGridY, xib, yib, Lam_Y, delta_y, cut)
    offset = offset + N_V
    res[offset:offset + N_P] = D_x @ U + D_y @ V
    offset = offset + N_P
    CAinv_x = interpPhi_x(UGridX, UGridY, xib, yib, U, delta_x, cut)
    CAinv_y = interpPhi_y(VGridX, VGridY, xib, yib, V, delta_y, cut)
    C_applied = interleave(CAinv_x, CAinv_y)  # now matches z_bc's ordering

    res[offset:offset + 2*Nib] = C_applied - K_res
    offset = offset + 2 * Nib
    # res[offset:offset + Nib] = interpPhi_x(UGridX, UGridY, xib, yib, U, delta_x, cut) - K_res[:Nib]
    # offset = offset + Nib
    # res[offset:offset + Nib] = interpPhi_y(VGridX, VGridY, xib, yib, V, delta_y, cut) - K_res[Nib:]
    # offset = offset + Nib
    res[offset:] = K.T @ Lam
    
    return res

def SchurLinearOperator_K(shape, lu_factorization, UGridX, UGridY, VGridX, VGridY,
                            xib, yib, center_x, center_y,
                            delta_x, delta_y,
                            N_U, N_V, N_P, Nib, cut):

    def mv(vec):
        Lam = vec[:2 * Nib]
        V = vec[2*Nib:]
        return LHS_op_big_K(Lam, V, lu_factorization, UGridX, UGridY, VGridX, VGridY, xib, yib, center_x, center_y, delta_x, delta_y, N_U, N_V, N_P, Nib, cut)
        
    return LinearOperator((shape, shape), matvec=mv)

def apply_Ainv_Stokes(rhs, lu_factorization, N_U, N_V):

    x = lu_factorization.solve(rhs)

    u = x[:N_U]
    v = x[N_U:N_U+N_V]
    p = x[N_U+N_V:]
    
    return u, v, p  

def compute_U_V_P_postprocessing_K(Lam_x, Lam_y, lu_factorization,
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

def build_staggered_Laps_do_nothing(Nx, Ny, dx, dy):
    dx2 = dx**2 
    dy2 = dy**2

    # LAP U: do-nothing in x, zero neumann in y (north) (U)

    e_x_U = np.ones(Nx + 2)
    e_y_U = np.ones(Ny + 1)
    D2_x_U = diags([e_x_U, -2*e_x_U, e_x_U], offsets=[-1, 0, 1], shape=(Nx + 2, Nx + 2), format='lil')
    D2_x_U[0, 1] = 2.0
    D2_x_U[-1, -2] = 2.0
    D2_x_U = (D2_x_U / dx2).tocsr()

    D2_y_U = diags([e_y_U, -2*e_y_U, e_y_U], offsets=[-1, 0, 1], shape=(Ny + 1, Ny + 1), format='lil')
    D2_y_U[0, 0] = -3.0
    D2_y_U[-1, -1] = -1.0
    D2_y_U = (D2_y_U / dy2).tocsr() 

    Lap_U = kron(eye(Nx + 2, format='csr'), D2_y_U, format='csr') + kron(D2_x_U, eye(Ny + 1, format='csr'), format='csr')

    # LAP V: do-nothing in y (north), zero neumman in x (V)

    e_x_V = np.ones(Nx + 1)
    e_y_V = np.ones(Ny + 2)
    D2_y_V = diags([e_y_V, -2*e_y_V, e_y_V], offsets=[-1, 0, 1], shape=(Ny + 2, Ny + 2), format='lil')
    # this is what we'll actually do with the divergence-free thing
    D2_y_V[-1, -2] = 2.0
    # D2_y_V[-2, -1] = 2.0
    D2_y_V[0, 1] = 0.0
    # print(D2_y_V.toarray())
    D2_y_V = (D2_y_V / dy2).tocsr() #/ dy2

    D2_x_V = diags([e_x_V, -2*e_x_V, e_x_V], offsets=[-1, 0, 1], shape=(Nx + 1, Nx + 1), format='lil')
    D2_x_V[0, 0] = -1.0
    D2_x_V[-1, -1] = -1.0
    D2_x_V = (D2_x_V / dx2).tocsr() #/ dx2

    Lap_V = kron(eye(Nx + 1, format='csr'), D2_y_V, format='csr') + kron(D2_x_V, eye(Ny + 2, format='csr'), format='csr')

    return Lap_U, Lap_V

def build_staggered_Grads_Divs_do_nothing(Nx, Ny, dx, dy):
    e_x_P = np.ones(Nx + 1)
    e_y_P = np.ones(Ny + 1)
    D2_x_P = diags([-e_x_P, e_x_P], offsets=[0, 1], shape=(Nx + 1, Nx + 2), format='lil')
    D2_x_P[0, 0] = -2.0
    D2_x_P[-1, -1] = 2.0
    D2_x_P = (D2_x_P / dx).tocsr() # / dx

    GT_x = kron(D2_x_P, eye(Nx + 1, format='csr'), format='csr')

    e_y_P = np.ones(Nx + 1)
    e_y_P = np.ones(Ny + 1)
    D2_y_P = diags([-e_y_P, e_y_P], offsets=[0, 1], shape=(Ny + 1, Ny + 2), format='lil')
    D2_y_P[0, 0] = 0.0
    D2_y_P[-1, -1] = 2.0
    D2_y_P = (D2_y_P / dy).tocsr() #/ 2 * dy

    GT_y = kron(eye(Ny + 1, format='csr'), D2_y_P, format='csr')

    G_x = (-GT_x.transpose()).tocsr()
    G_y = (-GT_y.transpose()).tocsr()

    # divergence operators do not include BCs 
    e_x_P = np.ones(Nx + 1)
    e_y_P = np.ones(Ny + 1)
    D2_x_P = diags([-e_x_P, e_x_P], offsets=[0, 1], shape=(Nx + 1, Nx + 2), format='lil')
    D2_x_P = (D2_x_P / dx).tocsr()

    D_x = kron(D2_x_P, eye(Nx + 1, format='csr'), format='csr')

    e_y_P = np.ones(Nx + 1)
    e_y_P = np.ones(Ny + 1)
    D2_y_P = diags([-e_y_P, e_y_P], offsets=[0, 1], shape=(Ny + 1, Ny + 2), format='lil')
    D2_y_P = (D2_y_P / dy).tocsr() # TODO: enforce V = 0 at the boundary? 

    D_y = kron(eye(Ny + 1, format='csr'), D2_y_P, format='csr')

    return G_x, G_y, D_x, D_y

# @jit(nopython=True)
def solve_from_svd(U, Sigma, Vh, rhs):
    Sigma_inv_diag = 1 / Sigma

    Sigma_inv = np.diag(Sigma_inv_diag)

    res = Vh.conj().T @ Sigma_inv @ U.T @ rhs

    return res

def solve_factorized_K(UGridX, UGridY, VGridX, VGridY, lu_factorization, U_schur, Sigma_schur, Vh_schur, f_bc, g_bc, h_bc, z_bc, V_bc, xib, yib, Nib, Nx, Ny, dx, tol, cut):
    delta_x, delta_y = make_composite_deltas(dx, n=3)

    N_U = (Nx + 2) * (Ny + 1)
    N_V = (Nx + 1) * (Ny + 2)
    N_P = (Nx + 1) * (Ny + 1)

    RHS = np.concatenate([f_bc, g_bc, h_bc, z_bc, V_bc])
    RHS_schur = RHS_op_big_K(RHS, lu_factorization, UGridX, UGridY, VGridX, VGridY, delta_x, delta_y, xib, yib, N_U, N_V, N_P, Nib, cut)

    # Solve
    sol = solve_from_svd(U_schur, Sigma_schur, Vh_schur, RHS_schur)
    
    # Split (no change in ordering of partition)
    lam_xy = sol[:2*Nib]
    V_rigid = sol[2*Nib:]

    # deinterleave
    lam_X = lam_xy[0::2]
    lam_Y = lam_xy[1::2]

    # Postprocessing: compute U and V
    U, V, P = compute_U_V_P_postprocessing_K(lam_X, lam_Y, lu_factorization, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, f_bc, g_bc, h_bc, N_U, N_V, cut)

    # # COMPUTE USING FULL OPERATOR CHECK
    # Nu = apply_A(np.concatenate([U, V, P, lam_X, lam_Y]), UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y, cut)
    # residual_check_N = np.linalg.norm(Nu - RHS) / np.linalg.norm(RHS)
    # print(f'residual Nu = {residual_check_N}')

    return U, V, P, lam_X, lam_Y, V_rigid

def solve_K(left_bound, right_bound, lu_factorization, f_bc, g_bc, h_bc, z_bc, V_bc, xib, yib, center_x, center_y, Nib, Nx, tol, cut):
    x = np.linspace(left_bound, right_bound, Nx + 2)
    dx = x[1] - x[0]
    y = x.copy()
    dy = dx

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

    RHS = np.concatenate([f_bc, g_bc, h_bc, z_bc, V_bc])
    RHS_schur = RHS_op_big_K(RHS, lu_factorization, UGridX, UGridY, VGridX, VGridY, delta_x, delta_y, xib, yib, N_U, N_V, N_P, Nib, cut) #schur_rhs_new(RHS, lu_factorization, UGridX, UGridY, VGridX, VGridY, delta_x, delta_y, xib, yib, N_U, N_V, N_P, Nib, cut)

    # Solve
    shape = 2 * Nib + 3
    SchurOp = SchurLinearOperator_K(shape, lu_factorization, UGridX, UGridY, VGridX, VGridY, xib, yib, center_x, center_y, delta_x, delta_y, N_U, N_V, N_P, Nib, cut)
    sol, _ = gmres(SchurOp, RHS_schur, rtol=tol, restart=500, callback=lambda rk: print(f"GMRES residual: {np.linalg.norm(rk)}"))

    # also check the actual residual directly, independent of what GMRES reports
    actual_resid = np.linalg.norm(SchurOp @ sol - RHS_schur) / np.linalg.norm(RHS_schur)
    print(f"Actual relative residual of Schur solve: {actual_resid}")

    # Split (no change in ordering of partition)
    lam_X = sol[:Nib]
    lam_Y = sol[Nib:2*Nib]
    V_rigid = sol[2*Nib:]

    # Postprocessing: compute U and V
    U, V, P = compute_U_V_P_postprocessing_K(lam_X, lam_Y, lu_factorization, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, f_bc, g_bc, h_bc, N_U, N_V, cut)

    return U, V, P, lam_X, lam_Y, V_rigid

def interleave(a, b):
    out = np.empty(a.size + b.size, dtype=a.dtype)
    out[0::2] = a
    out[1::2] = b
    return out

def RHS_op_big_K(rhs, lu_factorization,
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
    rhs_Jxy = rhs[offset:-3]
    rhs_V_rigid  = rhs[-3:]
    
    # deinterleave
    rhs_Jx = rhs_Jxy[0::2]
    rhs_Jy = rhs_Jxy[1::2]

    # STEP 1: Apply A^{-1} to (rhs_U, rhs_V, rhs_div)
    U, V, _ = apply_Ainv_Stokes(np.concatenate([rhs_U, rhs_V, rhs_div]), lu_factorization, N_U, N_V)

    # STEP 2: Apply C to the result
    CAinv_x = interpPhi_x(UGridX, UGridY, xib, yib, U, delta_x, cut)
    CAinv_y = interpPhi_y(VGridX, VGridY, xib, yib, V, delta_y, cut)
    CAinv_V = np.zeros(3)

    # STEP 3: Schur RHS = -C*A^{-1}(rhs)  +  original IB rhs components
    schur_rhs_x = -CAinv_x + rhs_Jx
    schur_rhs_y = -CAinv_y + rhs_Jy
    schur_rhs_V = -CAinv_V + rhs_V_rigid

    return np.concatenate((interleave(schur_rhs_x, schur_rhs_y), schur_rhs_V))

def LHS_op_big_K(Lam, V, lu_factorization,
                    UGridX, UGridY, VGridX, VGridY,
                    xib, yib, center_x, center_y,
                    delta_x, delta_y,
                    N_U, N_V, N_P, Nib, cut):
    K = build_K(xib, yib, [center_x, center_y], Nib)

    # ROADMAP: (D - CA^{-1}B)y = Dy - CA^{-1}By

    # B * [Lam_x, Lam_y]
    Lam_x = Lam[0::2]
    Lam_y = Lam[1::2]
    rhs_U = spreadQ_x(UGridX, UGridY, xib, yib, Lam_x, delta_x, cut)
    rhs_V = spreadQ_y(VGridX, VGridY, xib, yib, Lam_y, delta_y, cut)
    rhs_div = np.zeros(N_P) 

    # Solve A^{-1}
    U_solved, V_solved, _ = apply_Ainv_Stokes(np.concatenate([rhs_U, rhs_V, rhs_div]), lu_factorization, N_U, N_V)

    # C * (result)
    res_x = interpPhi_x(UGridX, UGridY, xib, yib, U_solved, delta_x, cut)
    res_y = interpPhi_y(VGridX, VGridY, xib, yib, V_solved, delta_y, cut)
    res_V = np.zeros(3)

    CAinv_B = interleave(res_x, res_y)

    # D * y
    # build D: 
    zeros_D_NW = np.zeros((2 * Nib, 2 * Nib))
    zeros_D_SE = np.zeros((3, 3))

    D_row_1 = np.hstack((zeros_D_NW, -K))
    D_row_2 = np.hstack((K.T, zeros_D_SE))
    D = np.vstack((D_row_1, D_row_2))

    D_applied = D @ np.concatenate([Lam, V])

    return D_applied - np.concatenate([CAinv_B, res_V])