
import numpy as np
import pyvista as pv
import cmasher as cmr
import cProfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from numba import jit
from sksparse.cholmod import cholesky
from scipy.sparse import spdiags, eye, kron, diags, csr_matrix, bmat, lil_matrix
from scipy.sparse.linalg import splu, eigs, spsolve, gmres, LinearOperator
from scipy.linalg import qr, lstsq
from scipy.io import loadmat, savemat
from scipy.interpolate import Akima1DInterpolator, interpn
import CPEO_utils_neumann as cpeo
import stokes_solver_utils_fast_K as stokes
from matplotlib.colors import ListedColormap, Normalize
import stokes_solver_utils_open_BCs as stokes

Nx_array = [512]
err_U_array = []
err_V_array = []
err_P_array = []

for Nx in Nx_array:
    ###########################
    ######  PARAMETERS  #######
    ###########################

    shear_velocity = 20

    # ## Grid parameters
    # Nx = 512 # 256; % number of grid points along one direction
    L = 4
    x = np.linspace(-L, L, Nx+2) 
    y = x.copy()
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    dx2 = dx**2
    dy2 = dy**2

    ##########################
    ######  GRID SETUP  ######
    ##########################

    ## Nodal grid (interior points)
    xint = x[1:-1]
    yint = y[1:-1]
    Ny = len(yint)

    X, Y = np.meshgrid(x, y)
    # Xint, Yint = np.meshgrid(xint, yint)

    ## Staggered grid (horizontal faces for V, vertical faces for U, cell centers for P)
    N_U = (Nx + 2) * (Ny + 1)
    N_V = (Nx + 1) * (Ny + 2)
    N_P = (Nx + 1) * (Ny + 1)
    x_mid = x + dx / 2
    y_mid = y + dy / 2
    x_offset = x_mid[:-1]
    y_offset = y_mid[:-1]

    Xint, Yint = np.meshgrid(x_offset, y_offset)

    UGridX, UGridY = np.meshgrid(x, y_offset) # UGridX, UGridY
    VGridX, VGridY = np.meshgrid(x_offset, y)
    PGridX, PGridY = np.meshgrid(x_offset, y_offset)

    #########################
    ######  OPERATORS  ######
    #########################

    ## first, test just the laplacian operators for U and V. 

    Lap_U, Lap_V = stokes.build_staggered_Laps_do_nothing(Nx, Ny, dx, dy)

    # next, test the grad operators for pi

    G_x, G_y, D_x, D_y = stokes.build_staggered_Grads_Divs_do_nothing(Nx, Ny, dx, dy)

    # compute ghost (forcing) values
    f_bc = np.zeros((Ny+1, Nx+2))
    f_bc[0, :] = -2 * shear_velocity / dy2

    ## Prefactor big Stokes operator
    Z_UV = csr_matrix((N_U, N_V))
    Z_VU = csr_matrix((N_V, N_U))
    Z_PP = csr_matrix((N_P, N_P))

    # Saddle point system
    big_L = bmat([
        [Lap_U, Z_VU,  -G_x],
        [Z_UV,  Lap_V, -G_y],
        [D_x,   D_y,   Z_PP] 
    ], format='csr')

    stokes_LU = splu(big_L)
    rhs_U = f_bc.ravel(order='F')
    rhs_V = np.zeros(N_V)
    rhs_P = np.zeros(N_P)

    res = stokes_LU.solve(np.concatenate([rhs_U, rhs_V, rhs_P]))

    U_solved, V_solved, P_solved = res[:N_U], res[N_U:N_U+N_V], res[N_U+N_V:]
    U_solved = U_solved.reshape((Ny+1, Nx+2), order='F')
    V_solved = V_solved.reshape((Ny+2, Nx+1), order='F')
    P_solved = P_solved.reshape((Ny+1, Nx+1), order='F')
    V_solved[0, :] = 0

    cmap = plt.cm.spring
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(VGridX, VGridY, V_solved, cmap=cmap, edgecolor='none')
    ax.set_title("V solved")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(UGridX, UGridY, U_solved, cmap=cmap, edgecolor='none')
    ax.set_title("U solved")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(PGridX, PGridY, P_solved, cmap=cmap, edgecolor='none')
    ax.set_title("P solved")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    V_interpolated = 0.5 * (V_solved[:-1, :] + V_solved[1:, :])
    U_interpolated = 0.5 * (U_solved[:, :-1] + U_solved[:, 1:])

    fig, ax = plt.subplots(figsize=(7, 7), dpi=100, facecolor='white')
    ax.set_facecolor('white')
    ax.streamplot(Xint, Yint, U_interpolated, V_interpolated, color='black', density=2, linewidth=1, arrowsize=1.5, zorder=1)
    ax.set_aspect('equal')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()