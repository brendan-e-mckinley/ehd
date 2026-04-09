import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pyvista as pv
from scipy.interpolate import interp1d

def random_radial_curve(H=10, a=1.5, b=1.0, y_offset=0, n_ib=25, seed=None):
    """
    Generates a randomized elliptical radial curve r(t) as a sum of sinusoidal
    harmonics with randomized amplitudes and phases, then reconstructs x(t), y(t).

    Parameters
    ----------
    H    : int, number of harmonics (default 10)
    a    : float, semi-major axis of base ellipse (default 1.5)
    b    : float, semi-minor axis of base ellipse (default 1.0)
    y_offset : offset in the Y direction
    seed : optional int, random seed for reproducibility

    Returns
    -------
    t, r, r_ellipse, x, y : np.ndarrays of shape (101,)
    """
    rng = np.random.default_rng(seed)

    # Randomize amplitude and phase
    rho = rng.normal(size=H) * np.logspace(-0.5, -2.5, H)
    phi = rng.random(H) * 2 * np.pi

    # Base elliptical radius r_ellipse(t) = ab / sqrt((b cos t)^2 + (a sin t)^2)
    t = np.linspace(0, 2 * np.pi, n_ib)
    r_ellipse = (a * b) / np.sqrt((b * np.cos(t))**2 + (a * np.sin(t))**2)

    # Accumulate harmonic perturbations on top of elliptical base
    r = r_ellipse.copy()
    for h in range(1, H + 1):
        r += rho[h - 1] * np.sin(h * t + phi[h - 1])

    # Reconstruct x(t), y(t)
    x = r * np.cos(t)
    y = r * np.sin(t) + y_offset

    return t, r, r_ellipse, x, y


def hexagonal_lattice(r=1.0, n=5):
    """
    Generates a hexagonal lattice of points within a circle of radius r.

    Parameters
    ----------
    r : float, radius of the circular region (default 1.0)
    n : int, number of points along one axis (default 5)

    Returns
    -------
    x, y : np.ndarrays of shape (N,), coordinates of lattice points
    """
    # Hexagonal lattice spacing
    dx = r / n
    dy = dx * np.sqrt(3) / 2

    # Generate grid points
    x_list, y_list = [], []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x_i = i * dx + (j % 2) * (dx / 2)
            y_j = j * dy
            if np.sqrt(x_i**2 + y_j**2) <= r:
                x_list.append(x_i)
                y_list.append(y_j)

    return np.array(x_list), np.array(y_list)

def interior_mask(x_mesh, y_mesh, x, y):
    """
    Returns a boolean mask that is True for points inside a closed curve.

    Parameters
    ----------
    x_mesh, y_mesh : np.ndarrays of any shape, query point coordinates
    x, y           : np.ndarrays (N,), coordinates of the closed curve

    Returns
    -------
    mask : boolean np.ndarray, same shape as x_mesh/y_mesh
    """
    x_close = np.append(x, x[0])
    y_close = np.append(y, y[0])
    curve  = Path(np.column_stack([x_close, y_close]))
    points = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])
    mask   = curve.contains_points(points).reshape(x_mesh.shape)
    return mask


def make_3d_mesh(x_int, y_int, x_curve, y_curve, dz=None):
    from scipy.spatial import KDTree

    # --- Default dz to minimum interior point spacing ---
    if dz is None:
        tree = KDTree(np.column_stack([x_int, y_int]))
        dists, _ = tree.query(np.column_stack([x_int, y_int]), k=2)
        dz = np.min(dists[:, 1])

    z_bot = -dz / 2
    z_top =  dz / 2

    # --- Two z-layers of interior points ---
    interior_top = np.column_stack([x_int, y_int, np.full_like(x_int, z_top)])
    interior_bot = np.column_stack([x_int, y_int, np.full_like(x_int, z_bot)])

    # --- Resample boundary curve to equal arc-length spacing of dz ---
    # Close the curve by appending the first point
    xc = np.append(x_curve, x_curve[0])
    yc = np.append(y_curve, y_curve[0])

    # Compute cumulative arc length
    ds           = np.sqrt(np.diff(xc)**2 + np.diff(yc)**2)
    arc_lengths  = np.append(0, np.cumsum(ds))
    total_length = arc_lengths[-1]

    # --- Remove duplicate arc-length values before interpolating ---
    _, unique_idx = np.unique(arc_lengths, return_index=True)
    arc_lengths   = arc_lengths[unique_idx]
    xc            = xc[unique_idx]
    yc            = yc[unique_idx]

    # Number of equally spaced points to match dz spacing
    n_edge    = int(np.round(total_length / dz))
    s_uniform = np.linspace(0, total_length, n_edge, endpoint=False)

    # Clamp s_uniform to the valid interpolation range to avoid edge extrapolation
    s_uniform = np.clip(s_uniform, arc_lengths[0], arc_lengths[-1])

    x_edge = interp1d(arc_lengths, xc, kind='cubic')(s_uniform)
    y_edge = interp1d(arc_lengths, yc, kind='cubic')(s_uniform)

    edge_ring = (1+dz/2.0)*np.column_stack([x_edge, y_edge, np.zeros(n_edge)])

    ### --- Stack everything ---
    # points = np.vstack([interior_top, interior_bot, edge_ring])
    # labels = np.concatenate([
    #     np.zeros(len(interior_top), dtype=int),  # 0: interior top
    #     np.ones( len(interior_bot), dtype=int),  # 1: interior bottom
    #     np.full( len(edge_ring),  2, dtype=int), # 2: edge
    # ])
    points = np.vstack([interior_top, interior_bot])
    labels = np.concatenate([
        np.zeros(len(interior_top), dtype=int),  # 0: interior top
        np.ones( len(interior_bot), dtype=int),  # 1: interior bottom
        #np.full( len(edge_ring),  2, dtype=int), # 2: edge
    ])

    return points, labels, dz, n_edge


def compute_normals(points, labels):
    """
    Computes normals for all points:
      - interior top (label=0): purely +z [0, 0, +1]
      - interior bot (label=1): purely -z [0, 0, -1]
      - edge ring   (label=2): outward-pointing normal in x-y plane, computed
                               from the periodic central difference tangent vector

    Parameters
    ----------
    points : np.ndarray (N, 3)
    labels : np.ndarray (N,), 0=interior top, 1=interior bottom, 2=edge

    Returns
    -------
    normals : np.ndarray (N, 3)
    """
    normals = np.zeros_like(points)

    # Top face: +z
    normals[labels == 0, 2] = +1.0

    # Bottom face: -z
    normals[labels == 1, 2] = -1.0

    # Edge ring: compute outward normal from periodic central difference tangent
    edge_mask = labels == 2
    if not np.any(edge_mask):
        return normals
    
    ex = points[edge_mask, 0]
    ey = points[edge_mask, 1]

    # Periodic central difference tangent: t_i = (r_{i+1} - r_{i-1}) / 2
    # np.roll handles the periodic wraparound automatically
    tx = (np.roll(ex, -1) - np.roll(ex, 1)) / 2.0
    ty = (np.roll(ey, -1) - np.roll(ey, 1)) / 2.0

    # Normal is tangent rotated 90 degrees: n = [-ty, tx] or [ty, -tx]
    # Choose [ty, -tx] and then enforce outward orientation below
    nx = ty
    ny = -tx

    # Normalize
    n_norm = np.sqrt(nx**2 + ny**2)
    nx /= n_norm
    ny /= n_norm

    # Enforce outward orientation by checking against centroid direction
    cx = ex.mean()
    cy = ey.mean()
    radial_x = ex - cx
    radial_y = ey - cy
    outward  = (nx * radial_x + ny * radial_y)  # dot product with radial direction
    flip     = np.where(outward < 0, -1.0, 1.0) # flip any inward-pointing normals

    normals[edge_mask, 0] = nx * flip
    normals[edge_mask, 1] = ny * flip
    normals[edge_mask, 2] = 0.0

    return normals

def compute_normals_2d(x, y):
    # Compute tangents using central difference
    tx = np.zeros_like(x)
    ty = np.zeros_like(y)

    tx[1:-1] = (x[2:] - x[:-2]) / 2.0
    ty[1:-1] = (y[2:] - y[:-2]) / 2.0

    # For endpoints, use one-sided differences
    tx[0] = x[1] - x[0]
    ty[0] = y[1] - y[0]
    tx[-1] = x[-1] - x[-2]
    ty[-1] = y[-1] - y[-2]

    # Normal is tangent rotated 90 degrees: n = [ty, -tx]
    nx = ty
    ny = -tx

    # Normalize
    n_norm = np.sqrt(nx**2 + ny**2)
    nx /= n_norm
    ny /= n_norm

    return np.column_stack([nx, ny])

def makeChipMesh(H=15, a=2.0, b=1.0, n_hex=20, seed=42):
    t, r, r_ellipse, x, y = random_radial_curve(H=H, a=a, b=b, seed=seed)
    x_hex, y_hex = hexagonal_lattice(r=1.5 * max(a, b), n=n_hex)

    r_scale = np.random.uniform(0.25, 1.2)
    x *= r_scale
    y *= r_scale

    mask = interior_mask(x_hex, y_hex, x, y)
    x_int = x_hex[mask]
    y_int = y_hex[mask]

    points, labels, dz, n_edge = make_3d_mesh(x_int, y_int, x, y)
    normals = compute_normals(points, labels)

    return points, labels, normals, dz

def top_blob(nib):
    a, b = 0.25, 0.5
    Nharm = 3
    t, r, r_ellipse, x, y = random_radial_curve(H=Nharm, a=a, b=b, y_offset=0.6, n_ib=nib, seed=200)

    normals = compute_normals_2d(x, y)

    return x, y, normals

    # fig, ax = plt.subplots()
    # ax.plot(0, 0, 'ko')
    # ax.plot(a * np.cos(t), b * np.sin(t), 'k--', label=f'ellipse (a={a}, b={b})')
    # ax.plot(x, y, 'bo-', label='r(t) curve')
    # ax.quiver(x, y, normals[:, 0], normals[:, 1], color='red', label='normals', scale=20)

    # ax.set_xlabel('x(t)')
    # ax.set_ylabel('y(t)')
    # ax.set_aspect('equal')
    # ax.legend()
    # ax.set_title('Parametric curve x(t), y(t)')

    # plt.show()

def bottom_blob(nib):
    a, b = 0.25, 0.5
    Nharm = 3
    t, r, r_ellipse, x, y = random_radial_curve(H=Nharm, a=a, b=b, y_offset=-0.6, n_ib=nib, seed=600)

    normals = compute_normals_2d(x, y)

    return x, y, normals

    # fig, ax = plt.subplots()
    # ax.plot(0, 0, 'ko')
    # ax.plot(a * np.cos(t), b * np.sin(t), 'k--', label=f'ellipse (a={a}, b={b})')
    # ax.plot(x, y, 'bo-', label='r(t) curve')
    # ax.quiver(x, y, normals[:, 0], normals[:, 1], color='red', label='normals', scale=20)

    # ax.set_xlabel('x(t)')
    # ax.set_ylabel('y(t)')
    # ax.set_aspect('equal')
    # ax.legend()
    # ax.set_title('Parametric curve x(t), y(t)')

    # plt.show()

if __name__ == '__main__':
    nib = 25
    x_t, y_t, n_t = top_blob(nib)
    x_b, y_b, n_b = bottom_blob(nib)

    fig, ax = plt.subplots()
    ax.plot(0, 0, 'ko')
    ax.plot(x_t, y_t, 'bo-', label='top curve')
    ax.quiver(x_t, y_t, n_t[:, 0], n_t[:, 1], color='red', label='top normals', scale=20)
    ax.plot(x_b, y_b, 'bo-', label='bottom curve')
    ax.quiver(x_b, y_b, n_b[:, 0], n_b[:, 1], color='red', label='bottom normals', scale=20)

    ax.set_xlabel('x(t)')
    ax.set_ylabel('y(t)')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Blobs')

    plt.show()


# if __name__ == '__main__':
#     points, labels, normals, dz = makeChipMesh(H=15, a=2.0, b=1.0, n_hex=20, seed=42)

#     # --- PyVista plot ---
#     cloud = pv.PolyData(points)
#     cloud['labels']  = labels
#     cloud['normals'] = normals

#     plotter = pv.Plotter()
#     plotter.set_background('white')

#     # Create spheres of radius dz/2 for interior points
#     for pts, color, label in [
#         (points[labels == 0], 'steelblue', 'Interior top'),
#         (points[labels == 1], 'seagreen',  'Interior bottom'),
#     ]:
#         spheres = pv.PolyData(pts).glyph(
#             geom=pv.Sphere(radius=dz/2),
#             scale=False,
#             orient=False,
#         )
#         plotter.add_mesh(spheres, color=color, label=label)

#     # Edge ring points (red)
#     if np.any(labels == 2):
#         plotter.add_mesh(cloud.extract_points(labels == 2),
#                      color='tomato', point_size=8,
#                      render_points_as_spheres=True, label='Edge ring')
#         # Draw the edge ring as a spline
#         ring_pts = np.vstack([points[labels == 2], points[labels == 2][0]])
#         spline   = pv.Spline(ring_pts, n_edge * 5)
#         plotter.add_mesh(spline, color='tomato', line_width=2)

#     # Normal arrows
#     arrow_cloud          = pv.PolyData(points)
#     arrow_cloud['normals'] = normals
#     glyphs = arrow_cloud.glyph(orient='normals', scale=False, factor=0.15)
#     plotter.add_mesh(glyphs, color='black', label='Normals')

    

#     plotter.add_legend()
#     plotter.show_axes()
#     plotter.show()

# if __name__ == '__main__':
#     a, b = 2.0, 1.0
#     Nharm = 10
#     t, r, r_ellipse, x, y = random_radial_curve(H=Nharm, a=a, b=b, seed=None)
#     x_hex, y_hex = hexagonal_lattice(r=1.5*max(a,b), n=20)

#     r_scale = np.random.uniform(0.25, 1.2)
#     x *= r_scale
#     y *= r_scale

#     mask = interior_mask(x_hex, y_hex, x, y)
#     x_int = x_hex[mask]
#     y_int = y_hex[mask]


#     fig, ax = plt.subplots()
#     ax.plot(x_hex, y_hex, 'ko', label='full hexagonal lattice points', markersize=3)
#     ax.plot(x_int, y_int, 'ro', label='clipped hexagonal lattice points')
#     ax.plot(0, 0, 'ko')
#     ax.plot(a * np.cos(t), b * np.sin(t), 'k--', label=f'ellipse (a={a}, b={b})')
#     ax.plot(x, y, 'b-', label='r(t) curve')
#     ax.set_xlabel('x(t)')
#     ax.set_ylabel('y(t)')
#     ax.set_aspect('equal')
#     ax.legend()
#     ax.set_title('Parametric curve x(t), y(t)')

#     plt.show()