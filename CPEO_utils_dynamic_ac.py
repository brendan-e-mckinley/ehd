import numpy as np
from scipy.sparse import diags, kron, eye, csr_matrix, bmat
from scipy.sparse.linalg import spsolve, gmres, LinearOperator
from scipy.interpolate import BSpline
from numba import jit
from sksparse.cholmod import cholesky

def Phi_exact_ac(t, x, y):
    beta_BC = 7.94
    return beta_BC * np.cos(20 * np.pi * t) * y + 0 * x

def compute_surface_maxwell_stress(Phi, G_x, G_y, Nx, Ny, Nib, xib, yib, center_x, center_y, Jop):
    G_x_Phi = G_x @ Phi
    G_y_Phi = G_y @ Phi
    
    G_x_Phi_interpolated = Jop(G_x_Phi.reshape(Ny, Nx, order='F'))
    G_y_Phi_interpolated = Jop(G_y_Phi.reshape(Ny, Nx, order='F'))

    Lambda_E_x = np.zeros_like(xib)
    Lambda_E_y = np.zeros_like(yib)

    for i in range(Nib):
        nu_i = [(xib[i] - center_x), (yib[i] - center_y)]
        nu_i_normalized = nu_i / np.linalg.norm(nu_i)

        Grad_Phi_i = np.array([G_x_Phi_interpolated[i], G_y_Phi_interpolated[i]])
        #Sigma_E_i = np.outer(Grad_Phi_i, Grad_Phi_i) - (1/2) * np.linalg.norm(Grad_Phi_i)**2 * np.eye(2)
        #Lambda_E = Sigma_E_i @ nu_i_normalized
        Lambda_E = Grad_Phi_i*np.dot(nu_i_normalized, Grad_Phi_i) - (1/2) * np.linalg.norm(Grad_Phi_i)**2 * nu_i_normalized
        Lambda_E_x[i] = Lambda_E[0]
        Lambda_E_y[i] = Lambda_E[1]

    return Lambda_E_x, Lambda_E_y

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


def NxLinearOperator(shape, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y):
    n = shape
    def mv(unknowns):
        return apply_N(unknowns, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y)
    return LinearOperator((n, n), matvec=mv)

def apply_N(x, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, N_U, N_V, N_P, Nib, Lap_U, Lap_V, G_x, G_y, D_x, D_y):
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

def SchurLinearOperator_N(shape, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, Lap_U, Lap_V, G_x, G_y, D_x, D_y, N_P, Nib):
    n = shape
    def mv(unknowns):
        unknown_P = unknowns[:N_P]
        unknown_Lam_x = unknowns[N_P:N_P + Nib]
        unknown_Lam_y = unknowns[N_P + Nib:N_P + 2*Nib]
        return apply_Schur_N([unknown_P, unknown_Lam_x, unknown_Lam_y], UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, Lap_U, Lap_V, G_x, G_y, D_x, D_y)
    return LinearOperator((n, n), matvec=mv)

def apply_Ainv_N(Lap_U, Lap_V, target_vec):
    target_vec_1, target_vec_2 = target_vec

    # second and third blocks are straightforward
    result_vec_1 = spsolve(Lap_U, target_vec_1)
    result_vec_2 = spsolve(Lap_V, target_vec_2)

    return [result_vec_1, result_vec_2]

def apply_Schur_N(unknown_blocks, UGridX, UGridY, VGridX, VGridY, xib, yib, delta_x, delta_y, Lap_U, Lap_V, G_x, G_y, D_x, D_y):
    Pi, Lam_x, Lam_y = unknown_blocks

    # apply B 
    B_U = spreadQ_x(UGridX, UGridY, xib, yib, Lam_x, delta_x) - G_x @ Pi
    B_V = spreadQ_y(VGridX, VGridY, xib, yib, Lam_y, delta_y) - G_y @ Pi

    # apply A inverse 
    Ainv_B_U, Ainv_B_V = apply_Ainv_N(Lap_U, Lap_V, [B_U, B_V])

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

def schur_rhs_N(rhs, Lap_U, Lap_V, D_x, D_y, delta_x, delta_y, UGridX, UGridY, VGridX, VGridY, xib, yib, N_U, N_V, N_P, Nib):
    rhs_1 = rhs[0:N_U]
    offset = N_U
    rhs_2 = rhs[offset:offset + N_V]
    offset = offset + N_V
    rhs_3 = rhs[offset:offset + N_P]
    offset = offset + N_P
    rhs_4 = rhs[offset:offset + Nib]
    offset = offset + Nib
    rhs_5 = rhs[offset:offset + Nib]

    Ainv_U, Ainv_V = apply_Ainv_N(Lap_U, Lap_V, [rhs_1, rhs_2])
    CAinv_1 = D_x @ Ainv_U + D_y @ Ainv_V
    CAinv_2 = interpPhi_x(UGridX, UGridY, xib, yib, Ainv_U, delta_x)
    CAinv_3 = interpPhi_y(VGridX, VGridY, xib, yib, Ainv_V, delta_y)

    schur_rhs_1 = CAinv_1 - rhs_3
    schur_rhs_2 = CAinv_2 - rhs_4
    schur_rhs_3 = CAinv_3 - rhs_5

    return np.concatenate((schur_rhs_1, schur_rhs_2, schur_rhs_3))

# Spreading operator
@jit(nopython=True)
def spreadQ_prime(X, Y, xq, yq, n_x, n_y, q, delta_r, cut):
    Sq = np.zeros_like(X)
    Nq = len(q)

    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    Nx = X.shape[1]
    Ny = Y.shape[0]

    for k in range(Nq):
        xk = xq[k]
        yk = yq[k]
        
        i_min = max(int((xk - cut - X[0, 0]) / dx), 0)
        i_max = min(int((xk + cut - X[0, 0]) / dx) + 1, Nx)
        j_min = max(int((yk - cut - Y[0, 0]) / dy), 0)
        j_max = min(int((yk + cut - Y[0, 0]) / dy) + 1, Ny)
        
        X_local = X[j_min:j_max, i_min:i_max]
        Y_local = Y[j_min:j_max, i_min:i_max]
        
        Rk = np.sqrt((X_local - xk)**2 + (Y_local - yk)**2)
        
        mask = (Rk <= cut)
        
        n_dot_rhat = np.where(mask,
            (n_x[k] * (X_local - xk) + n_y[k] * (Y_local - yk)) / Rk,
            0.0
        )
        
        contribution = q[k] * n_dot_rhat * delta_r(Rk) * mask
        Sq[j_min:j_max, i_min:i_max] += contribution
    
    return Sq

# Interpolation operator
@jit(nopython=True)
def interpPhi_prime(X, Y, xq, yq, n_x, n_y, Phi, delta_r, cut):
    Jphi = np.zeros_like(xq)
    dx_loc = X[0, 1] - X[0, 0]
    dy_loc = Y[1, 0] - Y[0, 0]
    Ny, Nx = X.shape

    for k in range(len(xq)):
        xk, yk = xq[k], yq[k]
        nxk, nyk = n_x[k], n_y[k]

        # Find the grid region affected by this point
        i_min = max(int((xk - cut - X[0,0]) / dx_loc), 0)
        i_max = min(int((xk + cut - X[0,0]) / dx_loc) + 1, Nx)
        j_min = max(int((yk - cut - Y[0,0]) / dy_loc), 0)
        j_max = min(int((yk + cut - Y[0,0]) / dy_loc) + 1, Ny)

        # Extract local patch
        X_local = X[j_min:j_max, i_min:i_max]
        Y_local = Y[j_min:j_max, i_min:i_max]
        Phi_local = Phi[j_min:j_max, i_min:i_max]

        dx = X_local - xk
        dy = Y_local - yk
        R = np.sqrt(dx**2 + dy**2)
        mask = R <= cut

        n_dot_rhat = np.where(mask, (nxk * dx + nyk * dy) / R, 0.0)

        delta_vals = delta_r(R)
        contribution = Phi_local * n_dot_rhat * delta_vals * mask

        Jphi[k] = dx_loc * dy_loc * np.sum(contribution)

    return Jphi

# @jit(nopython=True)
# def spreadQ(X, Y, xq, yq, q, delta, cut):
#     Sq = np.zeros_like(X)
#     Nq = len(q)
    
#     dx = X[0, 1] - X[0, 0]
#     dy = Y[1, 0] - Y[0, 0]
#     Nx = X.shape[1]
#     Ny = Y.shape[0]

#     for k in range(Nq):
#         xk = xq[k]
#         yk = yq[k]
        
#         # Compute local window indices
#         i_min = max(int((xk - cut - X[0, 0]) / dx), 0)
#         i_max = min(int((xk + cut - X[0, 0]) / dx) + 1, Nx)
#         j_min = max(int((yk - cut - Y[0, 0]) / dy), 0)
#         j_max = min(int((yk + cut - Y[0, 0]) / dy) + 1, Ny)
        
#         # Extract local window
#         X_local = X[j_min:j_max, i_min:i_max]
#         Y_local = Y[j_min:j_max, i_min:i_max]
        
#         Rk = np.sqrt((X_local - xk)**2 + (Y_local - yk)**2)
#         mask = (Rk <= cut)
        
#         contribution = q[k] * delta(Rk) * mask
        
#         Sq[j_min:j_max, i_min:i_max] += contribution
    
#     return Sq

@jit(nopython=True)
def interpPhi(X, Y, xq, yq, Phi, delta, cut):
    Jphi = np.zeros_like(xq)
    Nx = len(X)
    Ny = len(Y)
    dx_loc = X[0, 1] - X[0, 0]
    dy_loc = Y[1, 0] - Y[0, 0]
    
    for k in range(len(xq)):
        xk, yk = xq[k], yq[k]

        # Find the grid region affected by this point
        i_min = max(int((xk - cut - X[0,0]) / dx_loc), 0)
        i_max = min(int((xk + cut - X[0,0]) / dx_loc) + 1, Nx)
        j_min = max(int((yk - cut - Y[0,0]) / dy_loc), 0)
        j_max = min(int((yk + cut - Y[0,0]) / dy_loc) + 1, Ny)

        # Extract local patch
        X_local = X[j_min:j_max, i_min:i_max]
        Y_local = Y[j_min:j_max, i_min:i_max]
        Phi_local = Phi[j_min:j_max, i_min:i_max]

        dx = X_local - xk
        dy = Y_local - yk
        R = np.sqrt(dx**2 + dy**2)
        mask = R <= cut

        delta_vals = delta(R)
        contribution = Phi_local * delta_vals * mask

        Jphi[k] = dx_loc * dy_loc * np.sum(contribution)

    return Jphi

#@jit(nopython=True)
def Grad_dot_Grad(t, Phi, N_pm, dx, dy, Nx, Ny, N_pm_BC, X, Y):
    Phi_BC = Phi_exact_ac(t, X, Y)

    Phi = Phi.reshape(Ny, Nx).T
    N_pm = N_pm.reshape(Ny, Nx).T

    Phi_BC_y = np.vstack([Phi_BC[0, 1:-1].reshape(1, -1), Phi, Phi_BC[-1, 1:-1].reshape(1, -1)])
    Phi_y = (0.5/dy) * (Phi_BC_y[2:, :] - Phi_BC_y[:-2, :])

    N_pm_BC_y = np.vstack([N_pm_BC[0, 1:-1].reshape(1, -1), N_pm, N_pm_BC[-1, 1:-1].reshape(1, -1)])
    N_pm_y = (0.5/dy) * (N_pm_BC_y[2:, :] - N_pm_BC_y[:-2, :])

    Phi_BC_x = np.hstack([Phi_BC[1:-1, 0].reshape(-1, 1), Phi, Phi_BC[1:-1, -1].reshape(-1, 1)])
    Phi_x = (0.5/dx) * (Phi_BC_x[:, 2:] - Phi_BC_x[:, :-2])

    N_pm_BC_x = np.hstack([N_pm_BC[1:-1, 0].reshape(-1, 1), N_pm, N_pm_BC[1:-1, -1].reshape(-1, 1)])
    N_pm_x = (0.5/dx) * (N_pm_BC_x[:, 2:] - N_pm_BC_x[:, :-2])

    G_d_G = N_pm_x * Phi_x + N_pm_y * Phi_y
    return G_d_G.ravel(order='F')

def Constrained_Lap(ctxt, ctxt_prev, dLap, delta_layer, Nx, Ny, Nib, Sop_prime, Jop_prime):
    A_x_Ctx = np.zeros_like(ctxt)
    
    sz = Nx * Ny
    Phi = ctxt[:sz]
    N_p = ctxt[sz:2*sz]
    N_m = ctxt[2*sz:3*sz]
    q_i = 3 * sz
    Q = ctxt[q_i:q_i+Nib]
    Q_p = ctxt[q_i+Nib:q_i+2*Nib]
    Q_m = ctxt[q_i+2*Nib:q_i+3*Nib]
    
    SQ = Sop_prime(Q)
    SQ_p = Sop_prime(Q_p)
    SQ_m = Sop_prime(Q_m)
    
    dl2 = delta_layer**2

    A_x_Ctx[:sz] = dl2 * Phi - dLap.solve_A(0.5*N_p - 0.5*N_m + SQ.ravel(order='F'))
    A_x_Ctx[sz:2*sz] = N_p - dLap.solve_A(SQ_p.ravel(order='F'))
    A_x_Ctx[2*sz:3*sz] = N_m - dLap.solve_A(SQ_m.ravel(order='F'))
    A_x_Ctx[q_i:q_i+Nib] = Jop_prime(Phi.reshape(Ny, Nx, order='F'))
    A_x_Ctx[q_i+Nib:q_i+2*Nib] = Jop_prime(N_p.reshape(Ny, Nx, order='F'))
    A_x_Ctx[q_i+2*Nib:q_i+3*Nib] = Jop_prime(N_m.reshape(Ny, Nx, order='F'))
    
    return A_x_Ctx

def apply_Schur_R(dLap, p_blocks, delta_layer, Nx, Ny, Sop_prime, Jop_prime):
    p, p_p, p_m = p_blocks

    # apply B 
    Bp = Sop_prime(p)
    Bp_p = Sop_prime(p_p)
    Bp_m = Sop_prime(p_m)

    Bp_flat = Bp.ravel(order='F')
    Bp_p_flat = Bp_p.ravel(order='F')
    Bp_m_flat = Bp_m.ravel(order='F')

    # apply A inverse 
    Ainv_Bp, Ainv_Bp_p, Ainv_Bp_m = apply_Ainv_R(dLap, [Bp_flat, Bp_p_flat, Bp_m_flat], delta_layer)

    # apply C 
    res_1 = delta_layer * Jop_prime(Ainv_Bp.reshape(int(Ny), int(Nx), order='F'))
    res_2 = Jop_prime(Ainv_Bp_p.reshape(int(Ny), int(Nx), order='F'))
    res_3 = Jop_prime(Ainv_Bp_m.reshape(int(Ny), int(Nx), order='F'))

    return [res_1, res_2, res_3]

def apply_Ainv_R(dLap, target_vec, delta_layer):
    target_vec_1, target_vec_2, target_vec_3 = target_vec

    dl2 = delta_layer**2

    # second and third blocks are straightforward
    result_vec_2 = -dLap.solve_A(target_vec_2.ravel(order='F'))
    result_vec_3 = -dLap.solve_A(target_vec_3.ravel(order='F'))

    # use these results to compute first block 
    rhs = target_vec_1.ravel(order='F') - 0.5 * result_vec_2 + 0.5 * result_vec_3
    result_vec_1 = -dLap.solve_A(rhs / dl2)

    return [result_vec_1, result_vec_2, result_vec_3]

def Build_RHS_rho_ac(t, ctxt, ctxt_BCs, ac_value, N_p_prev, N_m_prev, U, V, Lap, dLap, G_d_G, delta_layer, Nx, Ny, Nib, Jop, Jop_prime, dx, dt):
    b_Ctx = np.zeros_like(ctxt_BCs)
    
    sz = Nx * Ny
    q_i = 3 * sz
    Phi = ctxt[:sz]
    N_p = ctxt[sz:2*sz]
    N_m = ctxt[2*sz:3*sz]
    
    Phi_BC = ctxt_BCs[:sz]
    N_p_BC = ctxt_BCs[sz:2*sz]
    N_m_BC = ctxt_BCs[2*sz:3*sz]
    Q_BC = ctxt_BCs[q_i:q_i+Nib]
    Q_p_BC = ctxt_BCs[q_i+Nib:q_i+2*Nib]
    Q_m_BC = ctxt_BCs[q_i+2*Nib:q_i+3*Nib]

    Phi_reshaped = Phi.reshape((Ny, Nx), order='F')
    N_p_reshaped = N_p.reshape((Ny, Nx), order='F')
    N_m_reshaped = N_m.reshape((Ny, Nx), order='F')

    # build full matrices
    Phi_full = np.zeros((Ny+2, Nx+2))
    N_p_full = np.ones((Ny+2, Nx+2))
    N_m_full = np.ones((Ny+2, Nx+2))

    Phi_full[1:-1, 1:-1] = Phi_reshaped
    Phi_full[1:-1, 0] = Phi_reshaped[:, -1]
    Phi_full[1:-1, -1] = Phi_reshaped[:, -1]
    Phi_full[0, :] = -ac_value
    Phi_full[-1, :] = ac_value
    N_p_full[1:-1, 1:-1] = N_p_reshaped
    N_m_full[1:-1, 1:-1] = N_m_reshaped
    
    dl2 = delta_layer**2

    computed_lap = (-Lap) @ Phi
    computed_lap = computed_lap + Phi_BC
    
    alpha_p = 1
    alpha_m = 1
    # U_G_x_N_p = U * (G_x_full @ N_p_full.ravel(order='F'))
    # V_G_y_N_p = V * (G_y_full @ N_p_full.ravel(order='F'))
    # U_G_x_N_m = U * (G_x_full @ N_m_full.ravel(order='F'))
    # V_G_y_N_m = V * (G_y_full @ N_m_full.ravel(order='F'))
    
    # u_G_N_p = alpha_p * (U_G_x_N_p + V_G_y_N_p)
    # u_G_N_m = alpha_m * (U_G_x_N_m + V_G_y_N_m)

    dNdx_p = np.where(
        U.reshape((Ny, Nx), order='F') > 0,
        (N_p_full[1:-1, 1:-1] - N_p_full[1:-1, :-2]) / dx,
        (N_p_full[1:-1, 2:]   - N_p_full[1:-1, 1:-1]) / dx
    )

    dNdy_p = np.where(
        V.reshape((Ny, Nx), order='F') > 0,
        (N_p_full[1:-1, 1:-1] - N_p_full[:-2,  1:-1]) / dx,
        (N_p_full[2:,  1:-1]  - N_p_full[1:-1, 1:-1]) / dx
    )

    dNdx_m = np.where(
        U.reshape((Ny, Nx), order='F') > 0,
        (N_m_full[1:-1, 1:-1] - N_m_full[1:-1, :-2]) / dx,
        (N_m_full[1:-1, 2:]   - N_m_full[1:-1, 1:-1]) / dx
    )

    dNdy_m = np.where(
        V.reshape((Ny, Nx), order='F') > 0,
        (N_m_full[1:-1, 1:-1] - N_m_full[:-2,  1:-1]) / dx,
        (N_m_full[2:,  1:-1]  - N_m_full[1:-1, 1:-1]) / dx
    )

    adv_p = alpha_p * (U.reshape((Ny, Nx), order='F') * dNdx_p + V.reshape((Ny, Nx), order='F') * dNdy_p)
    adv_m = alpha_m * (U.reshape((Ny, Nx), order='F') * dNdx_m + V.reshape((Ny, Nx), order='F') * dNdy_m)

    dt_scaled = dt
    n_p_time_deriv = (alpha_p / dt_scaled) * (eye(sz) @ (N_p - N_p_prev))
    n_m_time_deriv = (alpha_m / dt_scaled) * (eye(sz) @ (N_m - N_m_prev))

    b_Ctx[:sz] =  -dLap.solve_A(-dl2 * Phi_BC)
    b_Ctx[sz:2*sz] =  -dLap.solve_A(-N_p * computed_lap + adv_p.ravel(order='F') - N_p_BC - G_d_G(t, Phi, N_p) + n_p_time_deriv)
    b_Ctx[2*sz:3*sz] =  -dLap.solve_A(N_m * computed_lap + adv_m.ravel(order='F') - N_m_BC + G_d_G(t, Phi, N_m) + n_m_time_deriv)
    b_Ctx[q_i:q_i+Nib] = Q_BC
    b_Ctx[q_i+Nib:q_i+2*Nib] = Q_p_BC - Jop(N_p.reshape(Ny, Nx, order='F')) * Jop_prime(Phi.reshape(Ny, Nx, order='F'))
    b_Ctx[q_i+2*Nib:q_i+3*Nib] = Q_m_BC + Jop(N_m.reshape(Ny, Nx, order='F')) * Jop_prime(Phi.reshape(Ny, Nx, order='F'))
    
    return b_Ctx

def Build_RHS_Schur_System_ac(t, ctxt, ctxt_BCs, ac_value, N_p_prev, N_m_prev, U, V, Lap, G_d_G, delta_layer, Nx, Ny, Nib, Jop, Jop_prime, dx, dt):
    b_Ctx = np.zeros_like(ctxt_BCs)
    
    sz = Nx * Ny
    q_i = 3 * sz
    Phi = ctxt[:sz]
    N_p = ctxt[sz:2*sz]
    N_m = ctxt[2*sz:3*sz]
    
    Phi_BC = ctxt_BCs[:sz]
    N_p_BC = ctxt_BCs[sz:2*sz]
    N_m_BC = ctxt_BCs[2*sz:3*sz]
    Q_BC = ctxt_BCs[q_i:q_i+Nib]
    Q_p_BC = ctxt_BCs[q_i+Nib:q_i+2*Nib]
    Q_m_BC = ctxt_BCs[q_i+2*Nib:q_i+3*Nib]

    Phi_reshaped = Phi.reshape((Ny, Nx), order='F')
    N_p_reshaped = N_p.reshape((Ny, Nx), order='F')
    N_m_reshaped = N_m.reshape((Ny, Nx), order='F')

    # build full matrices
    Phi_full = np.zeros((Ny+2, Nx+2))
    N_p_full = np.ones((Ny+2, Nx+2))
    N_m_full = np.ones((Ny+2, Nx+2))

    Phi_full[1:-1, 1:-1] = Phi_reshaped
    Phi_full[1:-1, 0] = Phi_reshaped[:, -1]
    Phi_full[1:-1, -1] = Phi_reshaped[:, -1]
    Phi_full[0, :] = -ac_value
    Phi_full[-1, :] = ac_value
    N_p_full[1:-1, 1:-1] = N_p_reshaped
    N_m_full[1:-1, 1:-1] = N_m_reshaped

    dl2 = delta_layer**2

    computed_lap = (-Lap) @ Phi
    computed_lap = computed_lap + Phi_BC
    
    alpha_p = 1
    alpha_m = 1
    # U_G_x_N_p = U * (G_x_full @ N_p_full.ravel(order='F'))
    # V_G_y_N_p = V * (G_y_full @ N_p_full.ravel(order='F'))
    # U_G_x_N_m = U * (G_x_full @ N_m_full.ravel(order='F'))
    # V_G_y_N_m = V * (G_y_full @ N_m_full.ravel(order='F'))
    
    # u_G_N_p = alpha_p * (U_G_x_N_p + V_G_y_N_p)
    # u_G_N_m = alpha_m * (U_G_x_N_m + V_G_y_N_m)

    dNdx_p = np.where(
    U.reshape((Ny, Nx), order='F') > 0,
        (N_p_full[1:-1, 1:-1] - N_p_full[1:-1, :-2]) / dx,
        (N_p_full[1:-1, 2:]   - N_p_full[1:-1, 1:-1]) / dx
    )

    dNdy_p = np.where(
        V.reshape((Ny, Nx), order='F') > 0,
        (N_p_full[1:-1, 1:-1] - N_p_full[:-2,  1:-1]) / dx,
        (N_p_full[2:,  1:-1]  - N_p_full[1:-1, 1:-1]) / dx
    )

    dNdx_m = np.where(
        U.reshape((Ny, Nx), order='F') > 0,
        (N_m_full[1:-1, 1:-1] - N_m_full[1:-1, :-2]) / dx,
        (N_m_full[1:-1, 2:]   - N_m_full[1:-1, 1:-1]) / dx
    )

    dNdy_m = np.where(
        V.reshape((Ny, Nx), order='F') > 0,
        (N_m_full[1:-1, 1:-1] - N_m_full[:-2,  1:-1]) / dx,
        (N_m_full[2:,  1:-1]  - N_m_full[1:-1, 1:-1]) / dx
    )

    adv_p = alpha_p * (U.reshape((Ny, Nx), order='F') * dNdx_p + V.reshape((Ny, Nx), order='F') * dNdy_p)
    adv_m = alpha_m * (U.reshape((Ny, Nx), order='F') * dNdx_m + V.reshape((Ny, Nx), order='F') * dNdy_m)

    dt_scaled = dt
    n_p_time_deriv = (alpha_p / dt_scaled) * (eye(sz) @ (N_p - N_p_prev))
    n_m_time_deriv = (alpha_m / dt_scaled) * (eye(sz) @ (N_m - N_m_prev))

    b_Ctx[:sz] =  -dl2 * Phi_BC
    b_Ctx[sz:2*sz] =  -N_p * computed_lap + adv_p.ravel(order='F') - N_p_BC - G_d_G(t, Phi, N_p) + n_p_time_deriv
    b_Ctx[2*sz:3*sz] =  N_m * computed_lap + adv_m.ravel(order='F') - N_m_BC + G_d_G(t, Phi, N_m) + n_m_time_deriv
    b_Ctx[q_i:q_i+Nib] = Q_BC
    b_Ctx[q_i+Nib:q_i+2*Nib] = Q_p_BC - Jop(N_p.reshape(Ny, Nx, order='F')) * Jop_prime(Phi.reshape(Ny, Nx, order='F'))
    b_Ctx[q_i+2*Nib:q_i+3*Nib] = Q_m_BC + Jop(N_m.reshape(Ny, Nx, order='F')) * Jop_prime(Phi.reshape(Ny, Nx, order='F'))
    
    return b_Ctx

class ConstrainedLapOperator:
    def __init__(self, dLap, delta_layer, Nx, Ny, Nib, Sop_prime, Jop_prime):
        self.dLap = dLap
        self.delta_layer = delta_layer
        self.Nx = Nx
        self.Ny = Ny
        self.Nib = Nib
        self.Sop_prime = Sop_prime
        self.Jop_prime = Jop_prime
        self.ctxt_prev = None
    
    def set_context(self, ctxt_prev):
        self.ctxt_prev = ctxt_prev.copy()  # Make a copy to avoid reference issues
    
    def matvec(self, xx):
        return Constrained_Lap(xx, self.ctxt_prev, self.dLap, self.delta_layer, 
                              self.Nx, self.Ny, self.Nib, self.Sop_prime, self.Jop_prime)
    
# LinearOperator object for using the Schur complement as our LHS matrix in GMRES
def SchurLinearOperator_R(dLap, shape, Nib, Nx, Ny, delta_layer, Sop_prime, Jop_prime):
    n = shape
    def mv(p_block):
        p = p_block[0:Nib]
        p_p = p_block[Nib:2*Nib]
        p_m = p_block[2*Nib:3*Nib]

        res_1, res_2, res_3 = apply_Schur_R(dLap, [p, p_p, p_m], delta_layer, Nx, Ny, Sop_prime, Jop_prime)
        return np.concatenate([res_1, res_2, res_3])
    return LinearOperator((n, n), matvec=mv)

def schur_rhs_R(dLap, rhs, Nx, Ny, Nib, delta_layer, Jop_prime):
    sz = Nx * Ny
    q_i = 3 * sz

    rhs_1 = rhs[:sz]
    rhs_2 = rhs[sz:2*sz]
    rhs_3 = rhs[2*sz:3*sz]
    rhs_4 = rhs[q_i:q_i+Nib]
    rhs_5 = rhs[q_i+Nib:q_i+2*Nib]
    rhs_6 = rhs[q_i+2*Nib:q_i+3*Nib]

    Ainv_phi, Ainv_n_p, Ainv_n_m = apply_Ainv_R(dLap, [rhs_1, rhs_2, rhs_3], delta_layer)
    CAinv_phi = delta_layer * Jop_prime(Ainv_phi.reshape(Ny, Nx, order='F'))
    CAinv_n_p = Jop_prime(Ainv_n_p.reshape(Ny, Nx, order='F'))
    CAinv_n_m = Jop_prime(Ainv_n_m.reshape(Ny, Nx, order='F'))

    schur_rhs_1 = CAinv_phi - rhs_4
    schur_rhs_2 = CAinv_n_p - rhs_5
    schur_rhs_3 = CAinv_n_m - rhs_6

    return np.concatenate((schur_rhs_1, schur_rhs_2, schur_rhs_3))

def post_processing_compute_R(dLap, p_block, rhs, Nx, Ny, Nib, delta_layer, Sop_prime):
    sz = Nx * Ny

    p = p_block[0:Nib]
    p_p = p_block[Nib:2*Nib]
    p_m = p_block[2*Nib:3*Nib]

    rhs_1 = rhs[:sz]
    rhs_2 = rhs[sz:2*sz]
    rhs_3 = rhs[2*sz:3*sz]

    dl2 = delta_layer**2

    # get rhs for what we already know how to solve
    rhs_n_p = rhs_2 - Sop_prime(p_p).ravel(order='F')
    rhs_n_m = rhs_3 - Sop_prime(p_m).ravel(order='F')

    n_p = -dLap.solve_A(rhs_n_p)
    n_m = -dLap.solve_A(rhs_n_m)

    rhs_phi = (rhs_1 - 0.5*n_p + 0.5*n_m - Sop_prime(p).ravel(order='F')) / dl2
    phi = -dLap.solve_A(rhs_phi)

    return np.concatenate((phi, n_p, n_m, p, p_p, p_m))

@jit(nopython=True)
def solve_from_svd(U, Sigma, Vh, rhs):
    Sigma_inv_diag = 1 / Sigma

    Sigma_inv = np.diag(Sigma_inv_diag)

    res = Vh.conj().T @ Sigma_inv @ U.T @ rhs

    return res

# Check errors
# err2Norm_U = np.zeros(size)
# err2Norm_V = np.zeros(size)
# err2Norm_P = np.zeros(size)
# err2Norm_Lam_X = np.zeros(size)
# err2Norm_Lam_Y = np.zeros(size)
# for i in size_range:
#     err2Norm_U[i] = np.linalg.norm(UExactList[0:int(N[i]**2), i] - UNumericalList[0:int(N[i]**2), i], ord=2) / np.linalg.norm(UExactList[0:int(N[i]**2),i], ord=2)
#     err2Norm_V[i] = np.linalg.norm(VExactList[0:int(N[i]*(N[i]-1)), i] - VNumericalList[0:int(N[i]*(N[i]-1)), i], ord=2) / np.linalg.norm(VExactList[0:int(N[i]*(N[i]-1)),i], ord=2)
#     err2Norm_P[i] = np.linalg.norm(PExactList[0:int(N[i]**2), i] - PNumericalList[0:int(N[i]**2), i], ord=2) / np.linalg.norm(PExactList[0:int(N[i]**2),i], ord=2)
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
# plt.loglog(N, err2Norm_U, '--o', label=r'$Err_U$')
# plt.loglog(N, err2Norm_V, '--o', label=r'$Err_V$')
# plt.loglog(N, err2Norm_P, '--o', label=r'$Err_P$')
# plt.loglog(N, err2Norm_Lam_X, '--o', label=r'$Err_{\lambda_X}$')
# plt.loglog(N, err2Norm_Lam_Y, '--o', label=r'$Err_{\lambda_Y}$')
# plt.loglog(N, h**2, '--', label='2nd order')
