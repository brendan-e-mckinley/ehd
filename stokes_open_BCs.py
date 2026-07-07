
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

###########################
######  PARAMETERS  #######
###########################

## Grid parameters
Nx = 256 # 256; % number of grid points along one direction
L = 1
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
Xint, Yint = np.meshgrid(xint, yint)

## Staggered grid (horizontal faces for V, vertical faces for U, cell centers for P)
N_U = (Nx + 2) * (Ny + 1)
N_V = (Nx + 1) * (Ny + 2)
N_P = (Nx + 1) * (Ny + 1)
x_mid = x + dx / 2
y_mid = y + dy / 2
x_offset = x_mid[:-1]
y_offset = y_mid[:-1]

UGridX, UGridY = np.meshgrid(x, y_offset) # UGridX, UGridY
VGridX, VGridY = np.meshgrid(x_offset, y)
PGridX, PGridY = np.meshgrid(x_offset, y_offset)

#########################
######  OPERATORS  ######
#########################

## first, test just the laplacian operators for U and V. 

# do-nothing in y (north), zero neumman in x (V)

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

print(Lap_V.toarray())

# test with manufactured solution
def V_func(x, y):
    return (x**3 / 3 - x) * ((1 / 4) * (y + 1)**3 * (y - 3))
def Lap_V_func(x, y):
    return (2 * x) * ((1 / 4) * (y + 1)**3 * (y - 3)) + (x**3 / 3 - x) * (2 * (y + 1) * (y - 2) + (y + 1)**2)
def Div_V_y_func(x, y):
    return (x**3 / 3 - x) * (y + 1) ** 2 * (y - 2)

V_manufactured = V_func(VGridX, VGridY)
Lap_V_manufactured = Lap_V_func(VGridX, VGridY)
Div_V_manufactured = Div_V_y_func(PGridX, PGridY)

# compute ghost (forcing) values
g_bc = np.zeros_like(V_manufactured)
g_bc[-1, :] = -V_func(x_offset, L + dy) / dy2
#print(f_bc.ravel().T)

# solve
res = spsolve(Lap_V, Lap_V_manufactured.ravel(order='F') + g_bc.ravel(order='F'))

V_sol = res.reshape(Ny + 2, Nx + 1, order='F')

# solved values for the south boundary are garbage, we know they're equal to 0
V_sol[0, :] = 0

err = np.linalg.norm(V_sol - V_manufactured) / np.linalg.norm(V_manufactured)

print(f'error = {err}')

# do-nothing in x, zero neumann in y (north) (U)

e_x_U = np.ones(Nx + 2)
e_y_U = np.ones(Ny + 1)
D2_x_U = diags([e_x_U, -2*e_x_U, e_x_U], offsets=[-1, 0, 1], shape=(Nx + 2, Nx + 2), format='lil')
# this is what we'll actually do with the divergence-free thing
D2_x_U[0, 1] = 2.0
D2_x_U[-1, -2] = 2.0
# D2_x_U[1, 0] = 2.0
# D2_x_U[-2, -1] = 2.0
# print(D2_x_U.toarray())
D2_x_U = (D2_x_U / dx2).tocsr() #/ dx2

D2_y_U = diags([e_y_U, -2*e_y_U, e_y_U], offsets=[-1, 0, 1], shape=(Ny + 1, Ny + 1), format='lil')
D2_y_U[0, 0] = -3.0
D2_y_U[-1, -1] = -1.0
D2_y_U = (D2_y_U / dy2).tocsr() #/ dy2

Lap_U = kron(eye(Nx + 2, format='csr'), D2_y_U, format='csr') + kron(D2_x_U, eye(Ny + 1, format='csr'), format='csr')

# print(Lap_U.toarray())

# test with manufactured solution
def U_func(x, y):
    return (x**4 / 12 - x**2 / 2) * (y + 1)**2 * (y - 2)
def Lap_U_func(x, y):
    return (x**2 - 1) * (y + 1)**2 * (y - 2) + (x**4 / 12 - x**2 / 2) * (6 * y)
def Div_U_x_func(x, y):
    return (x**3 / 3 - x) * (y + 1)**2 * (y - 2)

U_manufactured = U_func(UGridX, UGridY)
Lap_U_manufactured = Lap_U_func(UGridX, UGridY)
Div_U_manufactured = Div_U_x_func(PGridX, PGridY)

# compute ghost (forcing) values
f_bc = np.zeros_like(U_manufactured)
f_bc[:, 0]  = -U_func(-dx, y_offset) / dx2
f_bc[:, -1] = -U_func(L + dx, y_offset) / dx2
#print(f_bc.ravel().T)

# solve
res = spsolve(Lap_U, Lap_U_manufactured.ravel(order='F') + f_bc.ravel(order='F'))

U_sol = res.reshape(Ny + 1, Nx + 2, order='F')

err = np.linalg.norm(U_sol - U_manufactured) / np.linalg.norm(U_manufactured)

print(f'error = {err}')

# next, test the grad operators for pi

e_x_P = np.ones(Nx + 1)
e_y_P = np.ones(Ny + 1)
D2_x_P = diags([-e_x_P, e_x_P], offsets=[0, 1], shape=(Nx + 1, Nx + 2), format='lil')
D2_x_P[0, 0] = -2.0
D2_x_P[-1, -1] = 2.0
# print(D2_x_P.toarray())
D2_x_P = (D2_x_P / dx).tocsr() # / dx

D_x = kron(D2_x_P, eye(Nx + 1, format='csr'), format='csr')
# print(D_x.toarray())

e_y_P = np.ones(Nx + 1)
e_y_P = np.ones(Ny + 1)
D2_y_P = diags([-e_y_P, e_y_P], offsets=[0, 1], shape=(Ny + 1, Ny + 2), format='lil')
D2_y_P[0, 0] = 0.0
D2_y_P[-1, -1] = 2.0
# print(D2_y_P.toarray())
D2_y_P = (D2_y_P / dy).tocsr() #/ 2 * dy

D_y = kron(eye(Ny + 1, format='csr'), D2_y_P, format='csr')
# print(D_y.toarray())

# Gradients are exact adjoints (negative transposes)
G_x = (-D_x.transpose()).tocsr()
G_y = (-D_y.transpose()).tocsr()

# different divergence operators? 
e_x_P = np.ones(Nx + 1)
e_y_P = np.ones(Ny + 1)
D2_x_P = diags([-e_x_P, e_x_P], offsets=[0, 1], shape=(Nx + 1, Nx + 2), format='lil')
D2_x_P = (D2_x_P / dx).tocsr() # / dx

D_x = kron(D2_x_P, eye(Nx + 1, format='csr'), format='csr')

e_y_P = np.ones(Nx + 1)
e_y_P = np.ones(Ny + 1)
D2_y_P = diags([-e_y_P, e_y_P], offsets=[0, 1], shape=(Ny + 1, Ny + 2), format='lil')
D2_y_P = (D2_y_P / dy).tocsr() #/ 2 * dy

D_y = kron(eye(Ny + 1, format='csr'), D2_y_P, format='csr')

print(G_x.toarray())
print(G_y.toarray())

# test with manufactured solution
def P_func(x, y):
    return (x ** 3 / 3 - x) * (y + 1)**2 * (y - 2)
def Grad_x_P_func(x, y):
    return (x ** 2 - 1) * (y + 1)**2 * (y - 2)
def Grad_y_P_func(x, y):
    return (x ** 3 / 3 - x) * (2 * (y + 1) * (y - 2) + (y + 1) ** 2)

P_manufactured = P_func(PGridX, PGridY)
Grad_x_P_manufactured = Grad_x_P_func(UGridX, UGridY)
Grad_y_P_manufactured = Grad_y_P_func(VGridX, VGridY)

# compute ghost (forcing) values
h_x_bc = np.zeros_like(Grad_x_P_manufactured)
h_x_bc[:, 0]  = U_func(-dx, y_offset) / dx2
h_x_bc[:, -1] = U_func(L + dx, y_offset) / dx2
h_y_bc = np.zeros_like(Grad_y_P_manufactured)
h_y_bc[-1, :] = -V_func(x_offset, L + dy) / dy2

# test
G_x_p = (G_x @ P_manufactured.ravel(order='F')).reshape(Ny + 1, Nx + 2, order='F')
G_y_p = (G_y @ P_manufactured.ravel(order='F')).reshape(Ny + 2, Nx + 1, order='F')

err_x = np.linalg.norm(G_x_p - Grad_x_P_manufactured) / np.linalg.norm(Grad_x_P_manufactured)
err_y = np.linalg.norm(G_y_p - Grad_y_P_manufactured) / np.linalg.norm(Grad_y_P_manufactured)

print(err_x)
print(err_y)

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
rhs_U = Lap_U_manufactured.ravel(order='F') - Grad_x_P_manufactured.ravel(order='F') #+ f_bc.ravel(order='F') + h_x_bc.ravel(order='F')
rhs_V = Lap_V_manufactured.ravel(order='F') - Grad_y_P_manufactured.ravel(order='F') #+ g_bc.ravel(order='F') + h_y_bc.ravel(order='F')
rhs_P = Div_U_manufactured.ravel(order='F') + Div_V_manufactured.ravel(order='F')

res = stokes_LU.solve(np.concatenate([rhs_U, rhs_V, rhs_P]))

U_solved, V_solved, P_solved = res[:N_U], res[N_U:N_U+N_V], res[N_U+N_V:]
V_sol[0, :] = 0

err_U = np.linalg.norm(U_solved - U_manufactured.ravel(order='F')) / np.linalg.norm(U_manufactured.ravel(order='F'))
err_V = np.linalg.norm(V_solved - V_manufactured.ravel(order='F')) / np.linalg.norm(V_manufactured.ravel(order='F'))
err_P = np.linalg.norm(P_solved - P_manufactured.ravel(order='F')) / np.linalg.norm(P_manufactured.ravel(order='F'))

print('Error U: ', err_U)
print('Error V: ', err_V)
print('Error P: ', err_P)

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(VGridX, VGridY, V_solved.reshape(Ny + 2, Nx + 1, order='F'), cmap=cmap, edgecolor='none')
ax.set_title("V solved")
ax.set_xlabel("x"); ax.set_ylabel("y")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(VGridX, VGridY, V_manufactured, cmap=cmap, edgecolor='none')
ax.set_title("V manufactured")
ax.set_xlabel("x"); ax.set_ylabel("y")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(VGridX, VGridY, V_solved.reshape(Ny + 2, Nx + 1, order='F') - V_manufactured, cmap=cmap, edgecolor='none')
ax.set_title("V error")
ax.set_xlabel("x"); ax.set_ylabel("y")

plt.show()