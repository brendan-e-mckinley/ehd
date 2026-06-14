import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

###########################
######  PARAMETERS  #######
###########################

## Grid parameters
Nx = 512 # 256; % number of grid points along one direction
L = 8.0 * np.pi 
x = np.linspace(-L/2, L/2, Nx+2) 
dx = x[1] - x[0]
y = x.copy()
dy = y[1] - y[0]

## Miscellaneous parameters
tol = 1e-4
beta_BC = 50 / L
sigma_bc = 0.78  # 0.68
delta_layer = 10 * dx #0.1#; %6*dx;
cut = 6 * 1.2 * dx # cutoff value

# Anderson acceleration parameters
beta = 0.2
m = 50

# time parameters
N_t = 1
dt = 0.01

##########################
######  GRID SETUP  ######
##########################

## Nodal grid (interior points)
xint = x[1:-1]
yint = y[1:-1]
Ny = len(yint)

X, Y = np.meshgrid(x, y)
Xint, Yint = np.meshgrid(xint, yint)

# For body_interpolated_x: shape (Ny+1, Nx), lives at y-midpoints, x-interior
x_mid_inner = x[1:-1]                  # length Nx
y_mid = 0.5 * (y[:-1] + y[1:])        # length Ny+1
Xplot_bx, Yplot_bx = np.meshgrid(x_mid_inner, y_mid)

# For body_interpolated_y: shape (Ny, Nx+1), lives at y-interior, x-midpoints
x_mid = 0.5 * (x[:-1] + x[1:])        # length Nx+1
y_mid_inner = y[1:-1]                  # length Ny
Xplot_by, Yplot_by = np.meshgrid(x_mid, y_mid_inner)

## Initial conditions for Rphi = rho system
ld = loadmat('maxwell_stress.mat')

body_interpolated_x = ld['body_x']
body_interpolated_y = ld['body_y']
phi = ld['Phi']

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xint, Yint, phi, cmap=cmap, edgecolor='none')
ax.set_title("phi")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.show()

# RECTANGULAR BOXES

bod_x_int_y_l = np.trapezoid(body_interpolated_x[int(Nx/16):int(5 * Nx/16),:], y_mid[int(Nx/16):int(5 * Nx/16)], axis=0)
final_integral_x_l = np.trapezoid(bod_x_int_y_l, x_mid_inner, axis=0)

bod_x_int_y_m = np.trapezoid(body_interpolated_x[int(Nx/16):int(5 * Nx/16),int(Nx/4):int(3 * Nx/4)], y_mid[int(Nx/16):int(5 * Nx/16)], axis=0)
final_integral_x_m = np.trapezoid(bod_x_int_y_m, x_mid_inner[int(Nx/4):int(3 * Nx/4)], axis=0)

bod_x_int_y_s = np.trapezoid(body_interpolated_x[int(Nx/16):int(5 * Nx/16),int(3 * Nx/8):int(5 * Nx/8)], y_mid[int(Nx/16):int(5 * Nx/16)], axis=0)
final_integral_x_s = np.trapezoid(bod_x_int_y_s, x_mid_inner[int(3 * Nx/8):int(5 * Nx/8)], axis=0)

bod_y_int_y_l = np.trapezoid(body_interpolated_y[int(Nx/16):int(5 * Nx/16),:], y_mid_inner[int(Nx/16):int(5 * Nx/16)], axis=0)
final_integral_y_l = np.trapezoid(bod_y_int_y_l, x_mid, axis=0)

bod_y_int_y_m = np.trapezoid(body_interpolated_y[int(Nx/16):int(5 * Nx/16),int(Nx/4):int(3 * Nx/4)], y_mid_inner[int(Nx/16):int(5 * Nx/16)], axis=0)
final_integral_y_m = np.trapezoid(bod_y_int_y_m, x_mid[int(Nx/4):int(3 * Nx/4)], axis=0)

bod_y_int_y_s = np.trapezoid(body_interpolated_y[int(Nx/16):int(5 * Nx/16),int(3 * Nx/8):int(5 * Nx/8)], y_mid_inner[int(Nx/16):int(5 * Nx/16)], axis=0)
final_integral_y_s = np.trapezoid(bod_y_int_y_s, x_mid[int(3 * Nx/8):int(5 * Nx/8)], axis=0)

print(f'Large domain x forces integral: {final_integral_x_l}')
print(f'Med domain x forces integral: {final_integral_x_m}')
print(f'Small domain x forces integral: {final_integral_x_s}')
print(f'Large domain y forces integral: {final_integral_y_l}')
print(f'Med domain y forces integral: {final_integral_y_m}')
print(f'Small domain y forces integral: {final_integral_y_s}')

## body forces plots

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xplot_bx[int(Nx/16):int(5 * Nx/16),:], Yplot_bx[int(Nx/16):int(5 * Nx/16),:], body_interpolated_x[int(Nx/16):int(5 * Nx/16),:], cmap=cmap, edgecolor='none')
ax.set_title("x body forces large")
ax.set_xlabel("x"); ax.set_ylabel("y")

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xplot_by[int(Nx/16):int(5 * Nx/16),:], Yplot_by[int(Nx/16):int(5 * Nx/16),:], body_interpolated_y[int(Nx/16):int(5 * Nx/16),:], cmap=cmap, edgecolor='none')
ax.set_title("y body forces large")
ax.set_xlabel("x"); ax.set_ylabel("y")

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xplot_bx[int(Nx/16):int(5 * Nx/16),int(Nx/4):int(3 * Nx/4)], Yplot_bx[int(Nx/16):int(5 * Nx/16),int(Nx/4):int(3 * Nx/4)], body_interpolated_x[int(Nx/16):int(5 * Nx/16),int(Nx/4):int(3 * Nx/4)], cmap=cmap, edgecolor='none')
ax.set_title("x body forces med")
ax.set_xlabel("x"); ax.set_ylabel("y")

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xplot_by[int(Nx/16):int(5 * Nx/16),int(Nx/4):int(3 * Nx/4)], Yplot_by[int(Nx/16):int(5 * Nx/16),int(Nx/4):int(3 * Nx/4)], body_interpolated_y[int(Nx/16):int(5 * Nx/16),int(Nx/4):int(3 * Nx/4)], cmap=cmap, edgecolor='none')
ax.set_title("y body forces med")
ax.set_xlabel("x"); ax.set_ylabel("y")

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xplot_bx[int(Nx/16):int(5 * Nx/16),int(3 * Nx/8):int(5 * Nx/8)], Yplot_bx[int(Nx/16):int(5 * Nx/16),int(3 * Nx/8):int(5 * Nx/8)], body_interpolated_x[int(Nx/16):int(5 * Nx/16),int(3 * Nx/8):int(5 * Nx/8)], cmap=cmap, edgecolor='none')
ax.set_title("x body forces small")
ax.set_xlabel("x"); ax.set_ylabel("y")

cmap = plt.cm.spring
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xplot_by[int(Nx/16):int(5 * Nx/16),int(3 * Nx/8):int(5 * Nx/8)], Yplot_by[int(Nx/16):int(5 * Nx/16),int(3 * Nx/8):int(5 * Nx/8)], body_interpolated_y[int(Nx/16):int(5 * Nx/16),int(3 * Nx/8):int(5 * Nx/8)], cmap=cmap, edgecolor='none')
ax.set_title("y body forces small")
ax.set_xlabel("x"); ax.set_ylabel("y")

plt.show()

# SQUARE BOXES

# bod_x_int_y_l = np.trapezoid(body_interpolated_x, y_mid, axis=0)
# final_integral_x_l = np.trapezoid(bod_x_int_y_l, x_mid_inner, axis=0)

# bod_x_int_y_m = np.trapezoid(body_interpolated_x[0:int(Nx/2),int(Nx/4):int(3 * Nx/4)], y_mid[0:int(Nx/2)], axis=0)
# final_integral_x_m = np.trapezoid(bod_x_int_y_m, x_mid_inner[int(Nx/4):int(3 * Nx/4)], axis=0)

# bod_x_int_y_s = np.trapezoid(body_interpolated_x[0:int(Nx/4),int(3 * Nx/8):int(5 * Nx/8)], y_mid[0:int(Nx/4)], axis=0)
# final_integral_x_s = np.trapezoid(bod_x_int_y_s, x_mid_inner[int(3 * Nx/8):int(5 * Nx/8)], axis=0)

# bod_y_int_y_l = np.trapezoid(body_interpolated_y, y_mid_inner, axis=0)
# final_integral_y_l = np.trapezoid(bod_y_int_y_l, x_mid, axis=0)

# bod_y_int_y_m = np.trapezoid(body_interpolated_y[0:int(Nx/2),int(Nx/4):int(3 * Nx/4)], y_mid_inner[0:int(Nx/2)], axis=0)
# final_integral_y_m = np.trapezoid(bod_y_int_y_m, x_mid[int(Nx/4):int(3 * Nx/4)], axis=0)

# bod_x_int_y_s = np.trapezoid(body_interpolated_y[0:int(Nx/4),int(3 * Nx/8):int(5 * Nx/8)], y_mid_inner[0:int(Nx/4)], axis=0)
# final_integral_y_s = np.trapezoid(bod_x_int_y_s, x_mid[int(3 * Nx/8):int(5 * Nx/8)], axis=0)

# print(f'Large domain x forces integral: {final_integral_x_l}')
# print(f'Med domain x forces integral: {final_integral_x_m}')
# print(f'Small domain x forces integral: {final_integral_x_s}')
# print(f'Large domain y forces integral: {final_integral_y_l}')
# print(f'Med domain y forces integral: {final_integral_y_m}')
# print(f'Small domain y forces integral: {final_integral_y_s}')

# ## body forces plots

# cmap = plt.cm.spring
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(Xplot_bx, Yplot_bx, body_interpolated_x, cmap=cmap, edgecolor='none')
# ax.set_title("x body forces large")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# cmap = plt.cm.spring
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(Xplot_by, Yplot_by, body_interpolated_y, cmap=cmap, edgecolor='none')
# ax.set_title("y body forces large")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# cmap = plt.cm.spring
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(Xplot_bx[0:int(Nx/2),int(Nx/4):int(3 * Nx/4)], Yplot_bx[0:int(Nx/2),int(Nx/4):int(3 * Nx/4)], body_interpolated_x[0:int(Nx/2),int(Nx/4):int(3 * Nx/4)], cmap=cmap, edgecolor='none')
# ax.set_title("x body forces med")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# cmap = plt.cm.spring
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(Xplot_by[0:int(Nx/2),int(Nx/4):int(3 * Nx/4)], Yplot_by[0:int(Nx/2),int(Nx/4):int(3 * Nx/4)], body_interpolated_y[0:int(Nx/2),int(Nx/4):int(3 * Nx/4)], cmap=cmap, edgecolor='none')
# ax.set_title("y body forces med")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# cmap = plt.cm.spring
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(Xplot_bx[0:int(Nx/4),int(3 * Nx/8):int(5 * Nx/8)], Yplot_bx[0:int(Nx/4),int(3 * Nx/8):int(5 * Nx/8)], body_interpolated_x[0:int(Nx/4),int(3 * Nx/8):int(5 * Nx/8)], cmap=cmap, edgecolor='none')
# ax.set_title("x body forces small")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# cmap = plt.cm.spring
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(Xplot_by[0:int(Nx/4),int(3 * Nx/8):int(5 * Nx/8)], Yplot_by[0:int(Nx/4),int(3 * Nx/8):int(5 * Nx/8)], body_interpolated_y[0:int(Nx/4),int(3 * Nx/8):int(5 * Nx/8)], cmap=cmap, edgecolor='none')
# ax.set_title("y body forces small")
# ax.set_xlabel("x"); ax.set_ylabel("y")

# plt.show()