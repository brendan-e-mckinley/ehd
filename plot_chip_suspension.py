import numpy as np
import pyvista as pv
import cmasher as cmr
import cProfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from numba import jit
from scipy.sparse import spdiags, eye, kron, diags, csr_matrix, bmat, lil_matrix
from scipy.sparse.linalg import splu, eigs, spsolve, gmres, LinearOperator
from scipy.linalg import qr, lstsq
from scipy.io import loadmat, savemat
from scipy.interpolate import Akima1DInterpolator, interpn
import CPEO_utils_dynamic as cpeo
import stokes_solver_utils_fast as stokes
import chip_pack as chip
from matplotlib.path import Path
from matplotlib.patches import PathPatch

## Grid parameters
Nx = 450  # 256; % number of grid points along one direction
L = 4.0 * np.pi 
x = np.linspace(-L/2, L/2, Nx+2) 
dx = x[1] - x[0]
y = x.copy()
dy = y[1] - y[0]

## Miscellaneous parameters
tol = 1e-4
beta_BC = 7.94
sigma_bc = -0.05 #0.78  # 0.68
delta_layer = 0.1  # 5*dx; %6*dx;
cut = 6 * 1.2 * dx # cutoff value

## Exact solutions
def Phi_exact(x, y):
    return beta_BC * y + 0 * x
def Npm_exact(x, y):
    return 0 * x + 1.0

## Nodal grid (interior points)
xint = x[1:-1]
yint = y[1:-1]
Ny = len(yint)

X, Y = np.meshgrid(x, y)
Xint, Yint = np.meshgrid(xint, yint)

# IB
xib, yib, n_x, n_y = chip.get_chip_suspension()
chips, discs = chip.make_chip_suspension()

x_specific_chips = []
y_specific_chips = []
x_specific_discs = []
y_specific_discs = []

for chip_inst in chips:
    x_specific_chips.append(chip_inst['x'])
    y_specific_chips.append(chip_inst['y'])
for disc in discs:
    x_specific_discs.append(disc['x'])
    y_specific_discs.append(disc['y'])

Nib = len(xib)

ld = loadmat('chip_suspension_close_hella_refined.mat')
METHOD = 'cubic'  # equivalent to 'makima' in MATLAB

Ny_ld = int(ld['Ny'][0, 0])
Nx_ld = int(ld['Nx'][0, 0])
#Nib_ld = int(ld['Nib'][0, 0])
sz = Ny_ld * Nx_ld

ctxt_ld = ld['ctxt_Rphi'].ravel(order='F')
Phi = ctxt_ld[:sz].reshape(Ny_ld, Nx_ld, order='F')
Np = ctxt_ld[sz:2*sz].reshape(Ny_ld, Nx_ld, order='F')
Nm = ctxt_ld[2*sz:3*sz].reshape(Ny_ld, Nx_ld, order='F')

UFull = ld['U_fluid']
VFull = ld['V_fluid']

# Test which (X, Y) grid points are inside
points = np.column_stack([X.ravel(), Y.ravel()])
points_int = np.column_stack([Xint.ravel(), Yint.ravel()])

# Build boundary path for each region
boundaries = []
paths = []
masks = []
masks_int = []
patches = []

for it, chip in enumerate(chips):
    boundary = np.column_stack([x_specific_chips[it], y_specific_chips[it]])
    boundary = np.vstack([boundary, boundary[0]])
    path = Path(boundary)
    mask = path.contains_points(points).reshape(X.shape)
    mask_int = path.contains_points(points_int).reshape(Xint.shape)
    patch = PathPatch(path, facecolor='gray', edgecolor='black', alpha=0.5)
    paths.append(path)
    boundaries.append(boundary)
    masks.append(mask)
    masks_int.append(mask_int)
    patches.append(patch)

for it, disc in enumerate(discs):
    boundary = np.column_stack([x_specific_discs[it], y_specific_discs[it]])
    boundary = np.vstack([boundary, boundary[0]])
    path = Path(boundary)
    mask = path.contains_points(points).reshape(X.shape)
    mask_int = path.contains_points(points_int).reshape(Xint.shape)
    patch = PathPatch(path, facecolor='black', edgecolor='black', alpha=0.5)
    paths.append(path)
    boundaries.append(boundary)
    masks.append(mask)
    masks_int.append(mask_int)
    patches.append(patch)

combined_mask = np.logical_or.reduce(masks)
combined_mask_int = np.logical_or.reduce(masks_int)

# Apply masks
U_masked = np.where(combined_mask, np.nan, UFull)
V_masked = np.where(combined_mask, np.nan, VFull)
Np_masked = np.where(combined_mask_int, np.nan, Np)
Phi_masked = np.where(combined_mask_int, np.nan, Phi)
Phi_dist_check = Phi_exact(Xint, Yint)
Phi_dist_masked = np.where(combined_mask_int, np.nan, Phi_dist_check)
Phi_dist = Phi_masked - Phi_dist_masked

# cmr.viola trimmed colormap
from matplotlib.colors import ListedColormap, Normalize
new_cmap = plt.cm.get_cmap('cmr.ocean', 256)
new_colors = new_cmap(np.linspace(0.4, 0.9, 256))
custom_cmap = ListedColormap(new_colors)

plt.rcParams['font.family'] = 'Helvetica'
fig, ax = plt.subplots(figsize=(10, 8))

# ── ion concentration ──────────────────────────────────────────────

c_map = ax.pcolormesh(
    Xint, Yint, Np_masked,
    cmap='RdBu_r', shading='auto',
    vmin=0.98, vmax=1.02, zorder=1
)
fig.colorbar(c_map, ax=ax, location='left', label=r'$N_p$', pad=0.12)

# ── velocity magnitude + flow streamlines ─────────────────────────

ax.streamplot(
    X, Y, U_masked, V_masked,
    color='black', density=3, linewidth=0.8, arrowsize=1.2, zorder=3
)

for patch in patches:
    ax.add_patch(patch)

# ── labels ────────────────────────────────────────────────────────────────────
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')
ax.grid(True, linewidth=0.4, alpha=0.4)

# ax.text(-1.8, 1.85, 'E-field / $N_p$',  fontsize=10, color='white', zorder=6)
# ax.text( 0.1, 1.85, 'flow / |velocity|', fontsize=10, color='white', zorder=6)

plt.tight_layout()
plt.show()