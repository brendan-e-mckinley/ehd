import numpy as np
import pyvista as pv
import cmasher as cmr
import cProfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from numba import jit
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags, eye, kron, diags, csr_matrix, bmat, lil_matrix
from scipy.sparse.linalg import splu, eigs, spsolve, gmres, LinearOperator
from scipy.linalg import qr, lstsq
from scipy.io import loadmat, savemat
from scipy.interpolate import Akima1DInterpolator, interpn
import CPEO_utils_fix as cpeo
import stokes_solver_utils_fast as stokes
from matplotlib.colors import ListedColormap, Normalize

## Lopsided grid

# ## Grid parameters
# Nx, Ny = 4, 4 # 256; % number of grid points along one direction
# L = 2 * np.pi 
# x = np.linspace(-L/2, L/2, Nx) 
# dx = x[1] - x[0]
# y = x.copy()
# dy = y[1] - y[0]

# ## Miscellaneous parameters
# tol = 1e-4
# beta_BC = 50 / L
# # sigma_bc = 0.78  # 0.68
# delta_layer = 5 * dx  # 5*dx; %6*dx;
# # cut = 6 * 1.2 * dx # cutoff value

# # Anderson acceleration parameters
# beta = 0.2
# m = 50

# # time parameters
# N_t = 1
# dt = 0.01

# ##########################
# ######  GRID SETUP  ######
# ##########################

# X, Y = np.meshgrid(x, y)

# # Nodal laplacian
# e_y = np.ones(Ny - 1) #(1/dy**2) * 
# D2_d_y = spdiags([e_y, -2*e_y, e_y], [-1, 0, 1], Ny-1, Ny-1)

# dok_mat_y = D2_d_y.todok()
# dok_mat_y[0, 1] *= 2
# D2_d_y = dok_mat_y.tocsr()
# print(D2_d_y.toarray())

# e_x = np.ones(Nx) #(1/dy**2) * 
# D2_d_x = spdiags([e_x, -2*e_x, e_x], [-1, 0, 1], Nx, Nx)

# dok_mat_x = D2_d_x.todok()
# dok_mat_x[0, 1] *= 2
# dok_mat_x[-1, -2] *= 2
# D2_d_x = dok_mat_x.tocsr()
# print(D2_d_x.toarray())

# I_nx = eye(Nx - 1)
# I_ny = eye(Ny)
# Lap = -(kron(I_ny, D2_d_y) + kron(D2_d_x, I_nx))
# print(Lap.toarray())

# dLap = cholesky(Lap) # Cholesky decomposition

# rhs = np.zeros([Nx - 1, Ny])
# rhs[0, :] = -2 * beta_BC / dx
# print(rhs.ravel(order='F'))

# phi = np.zeros_like(X)
# phi_solved = -dLap.solve_A(rhs.ravel(order='F'))
# phi_reshaped = phi_solved.reshape(Ny-1,Nx, order='F')
# phi[:-1,:] = phi_reshaped

# cmap = plt.cm.spring
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X, Y, phi, cmap=cmap, edgecolor='none')
# ax.set_title("Phi")
# ax.set_xlabel("x"); ax.set_ylabel("y")
# plt.show()

# ## PHI LAPLACIAN: Interior grid

# ## Grid parameters
# Nx, Ny = 128, 128 # 256; % number of grid points along one direction
# L = 2 * np.pi 
# x = np.linspace(-L/2, L/2, Nx + 2) 
# dx = x[1] - x[0]
# y = x.copy()
# dy = y[1] - y[0]

# xint = x[1:-1]
# yint = y[1:-1]

# ## Miscellaneous parameters
# tol = 1e-4
# beta_BC = 50 / L
# # sigma_bc = 0.78  # 0.68
# delta_layer = 5 * dx  # 5*dx; %6*dx;
# # cut = 6 * 1.2 * dx # cutoff value

# # Anderson acceleration parameters
# beta = 0.2
# m = 50

# # time parameters
# N_t = 1
# dt = 0.01

# ##########################
# ######  GRID SETUP  ######
# ##########################

# X, Y = np.meshgrid(x, y)
# Xint, Yint = np.meshgrid(xint, yint)

# # Nodal laplacian
# e_y = (1/dy**2) * np.ones(Ny) #(1/dy**2) * 
# D2_d_y = spdiags([e_y, -2*e_y, e_y], [-1, 0, 1], Ny, Ny)

# dok_mat_y = D2_d_y.todok()
# dok_mat_y[-1, -2] *= 2
# D2_d_y = dok_mat_y.tocsr()
# print(D2_d_y.toarray())

# e_x = (1/dy**2) * np.ones(Nx) #(1/dy**2) * 
# D2_d_x = spdiags([e_x, -2*e_x, e_x], [-1, 0, 1], Nx, Nx)

# dok_mat_x = D2_d_x.todok()
# dok_mat_x[0, 1] *= 2
# dok_mat_x[-1, -2] *= 2
# D2_d_x = dok_mat_x.tocsr()
# print(D2_d_x.toarray())

# I_nx = eye(Nx)
# I_ny = eye(Ny)
# Lap = -(kron(I_nx, D2_d_y) + kron(D2_d_x, I_ny))
# print(Lap.toarray())

# rhs = np.zeros([Nx, Ny])
# rhs[-1, :] = 2 * beta_BC / dx
# print(rhs.ravel(order='F'))

# phi_solved = spsolve(Lap, rhs.ravel(order='F'))
# phi = phi_solved.reshape(Ny,Nx, order='F')

# cmap = plt.cm.spring
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(Xint, Yint, phi, cmap=cmap, edgecolor='none')
# ax.set_title("Phi")
# ax.set_xlabel("x"); ax.set_ylabel("y")
# plt.show()

# computed_rhs = Lap @ phi.ravel(order='F')

# residual = computed_rhs - rhs.ravel(order='F')
# print(residual)

## N_PM LAPLACIAN: Interior grid

## Grid parameters
Nx, Ny = 128, 128 # 256; % number of grid points along one direction
L = 2 * np.pi 
x = np.linspace(-L/2, L/2, Nx + 2) 
dx = x[1] - x[0]
y = x.copy()
dy = y[1] - y[0]

xint = x[1:-1]
yint = y[1:-1]

## Miscellaneous parameters
tol = 1e-4
beta_BC = 50 / L
# sigma_bc = 0.78  # 0.68
delta_layer = 5 * dx  # 5*dx; %6*dx;
# cut = 6 * 1.2 * dx # cutoff value

# Anderson acceleration parameters
beta = 0.2
m = 50

# time parameters
N_t = 1
dt = 0.01

##########################
######  GRID SETUP  ######
##########################

X, Y = np.meshgrid(x, y)
Xint, Yint = np.meshgrid(xint, yint)

# Nodal laplacian
e_y = (1/dy**2) * np.ones(Ny) #(1/dy**2) * 
D2_d_y = spdiags([e_y, -2*e_y, e_y], [-1, 0, 1], Ny, Ny)

dok_mat_y = D2_d_y.todok()
D2_d_y = dok_mat_y.tocsr()
print(D2_d_y.toarray())

e_x = (1/dy**2) * np.ones(Nx) #(1/dy**2) * 
D2_d_x = spdiags([e_x, -2*e_x, e_x], [-1, 0, 1], Nx, Nx)

dok_mat_x = D2_d_x.todok()
dok_mat_x[0, 1] *= 2
dok_mat_x[-1, -2] *= 2
D2_d_x = dok_mat_x.tocsr()
print(D2_d_x.toarray())

I_nx = eye(Nx)
I_ny = eye(Ny)
Lap = -(kron(I_nx, D2_d_y) + kron(D2_d_x, I_ny))
print(Lap.toarray())

rhs = np.zeros([Nx, Ny])
rhs[-1, :] = 1 / (dx**2)
print(rhs.ravel(order='F'))

npm_solved = spsolve(Lap, rhs.ravel(order='F'))
npm = npm_solved.reshape(Ny,Nx, order='F')

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xint, Yint, npm, cmap=cmap, edgecolor='none')
ax.set_title("Npm")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.show()

computed_rhs = Lap @ npm.ravel(order='F')

residual = computed_rhs - rhs.ravel(order='F')
print(residual)