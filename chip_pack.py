import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path


# ── Arc-length resampling ─────────────────────────────────────────────────────

def resample_by_arc_length(x, y, ds):
    """
    Resample a closed curve (x, y) so that consecutive points are spaced
    approximately `ds` apart in arc length.

    The input curve is treated as closed: the segment from the last point
    back to the first is included in the total perimeter.  The resampled
    output also closes (last point ≠ first point, but the implicit closing
    edge has length ≈ ds).

    Parameters
    ----------
    x, y : array-like, original curve vertices (need not be uniformly spaced)
    ds   : float, target arc-length spacing

    Returns
    -------
    xr, yr : np.ndarray, resampled vertices
    """
    x, y = np.asarray(x, float), np.asarray(y, float)

    # Close the curve for arc-length computation
    xc = np.append(x, x[0])
    yc = np.append(y, y[0])

    # Cumulative chord length
    dx = np.diff(xc)
    dy = np.diff(yc)
    seg_len = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_length = s[-1]

    # Number of output points: perimeter / ds, rounded to nearest int >= 3
    n_out = max(3, round(total_length / ds))
    s_uniform = np.linspace(0.0, total_length, n_out, endpoint=False)

    xr = np.interp(s_uniform, s, xc)
    yr = np.interp(s_uniform, s, yc)
    return xr, yr


def circle_points(cx, cy, r, ds):
    """
    Discretize a circle of radius `r` centred at (cx, cy) with arc-length
    spacing `ds`.

    Returns
    -------
    x, y : np.ndarray of shape (n,)  — closed polygon vertices (last ≠ first)
    """
    circumference = 2.0 * np.pi * r
    n = max(3, round(circumference / ds))
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return cx + r * np.cos(t), cy + r * np.sin(t)


# ── Core curve generation ─────────────────────────────────────────────────────

def random_radial_curve(H=10, a=1.5, b=1.0, ds=0.1, n_init=2000, seed=None):
    """
    Generates a randomized elliptical radial curve, then resamples it so that
    consecutive vertices are spaced approximately `ds` apart in arc length.

    Parameters
    ----------
    H      : int, number of harmonics
    a, b   : float, semi-axes of the base ellipse
    ds     : float, target arc-length spacing for the output vertices
    n_init : int, number of points used for the initial (uniform-in-t) curve
             before resampling — should be large enough that no chord is >> ds
    seed   : optional int

    Returns
    -------
    x, y : np.ndarray — arc-length resampled curve vertices
    """
    rng = np.random.default_rng(seed)
    rho = rng.normal(size=H) * np.logspace(-0.5, -2.5, H)
    phi = rng.random(H) * 2 * np.pi

    t = np.linspace(0, 2 * np.pi, n_init, endpoint=False)
    r_ellipse = (a * b) / np.sqrt((b * np.cos(t))**2 + (a * np.sin(t))**2)

    r = r_ellipse.copy()
    for h in range(1, H + 1):
        r += rho[h - 1] * np.sin(h * t + phi[h - 1])

    x = r * np.cos(t)
    y = r * np.sin(t)

    # Resample to uniform arc-length spacing
    x, y = resample_by_arc_length(x, y, ds)
    return x, y


def compute_normals_2d(x, y):
    """Outward-pointing unit normals via central-difference tangent rotation."""
    # Treat as closed: wrap around for end-point differences
    xc = np.append(x, x[0])
    yc = np.append(y, y[0])

    tx = np.zeros(len(x))
    ty = np.zeros(len(x))
    # Central difference using the closed neighbours
    xext = np.concatenate([[x[-1]], x, [x[0]]])
    yext = np.concatenate([[y[-1]], y, [y[0]]])
    tx = (xext[2:] - xext[:-2]) / 2.0
    ty = (yext[2:] - yext[:-2]) / 2.0

    nx, ny = ty, -tx
    n_norm = np.sqrt(nx**2 + ny**2) + 1e-12
    return np.column_stack([nx / n_norm, ny / n_norm])


# ── Geometry helpers ──────────────────────────────────────────────────────────

def curve_bbox(x, y):
    return x.min(), x.max(), y.min(), y.max()


def expand_polygon(x, y, amount):
    """Offset every vertex outward along its unit normal by `amount`."""
    normals = compute_normals_2d(x, y)
    return x + normals[:, 0] * amount, y + normals[:, 1] * amount


def bboxes_overlap(b1, b2, gap=0.0):
    """True if the two bounding boxes are closer than `gap` in any axis."""
    xmin1, xmax1, ymin1, ymax1 = b1
    xmin2, xmax2, ymin2, ymax2 = b2
    return not (xmax1 + gap < xmin2 or xmax2 + gap < xmin1 or
                ymax1 + gap < ymin2 or ymax2 + gap < ymin1)


def polygons_collide(xy_new, xy_new_exp, xy_placed, xy_placed_exp):
    """
    Robust polygon-polygon collision test with gap baked into expanded polygons.
    Catches containment, boundary crossing, and near-miss within gap.
    """
    path_new     = Path(xy_new)
    path_new_exp = Path(xy_new_exp)
    path_pl      = Path(xy_placed)
    path_pl_exp  = Path(xy_placed_exp)
    step = max(1, len(xy_new) // 60)
    if path_pl.contains_points(xy_new_exp[::step]).any():     return True
    if path_new.contains_points(xy_placed_exp[::step]).any(): return True
    if path_pl_exp.contains_points(xy_new[::step]).any():     return True
    if path_new_exp.contains_points(xy_placed[::step]).any(): return True
    return False


def circle_bbox(cx, cy, r):
    return cx - r, cx + r, cy - r, cy + r


def circle_circle_collide(cx1, cy1, r1, cx2, cy2, r2, gap):
    """True if two discs are closer than `gap` boundary-to-boundary."""
    return np.hypot(cx1 - cx2, cy1 - cy2) < r1 + r2 + gap


def circle_polygon_collide(cx, cy, r, xy_poly, xy_poly_exp, gap):
    """
    True if a disc collides with a polygon respecting `gap`.
    Tests: centre inside expanded polygon, polygon vertex inside disc exclusion
    zone (r + gap/2), centre inside original polygon.
    """
    r_exp = r + gap / 2.0
    pt = np.array([[cx, cy]])
    if Path(xy_poly_exp).contains_points(pt)[0]:  return True
    if Path(xy_poly).contains_points(pt)[0]:       return True
    dists = np.hypot(xy_poly[:, 0] - cx, xy_poly[:, 1] - cy)
    if np.any(dists < r_exp):                      return True
    return False


# ── Proximity-biased disc centre sampler ──────────────────────────────────────

def sample_near_objects(placed_objects, rng, domain, disc_r, gap,
                        n_candidates=12, spread_factor=2.0):
    """
    Generate candidate disc centres biased to lie near already-placed objects.
    For curves: offset from a random boundary vertex along its outward normal.
    For discs:  offset from disc centre in a random direction.
    """
    lo, hi = domain
    margin = disc_r + gap / 2.0
    candidates = []
    for _ in range(n_candidates):
        obj = placed_objects[rng.integers(len(placed_objects))]
        min_offset = disc_r + gap
        if obj['type'] == 'curve':
            idx = rng.integers(len(obj['x']))
            bx, by = obj['x'][idx], obj['y'][idx]
            normals = compute_normals_2d(obj['x'], obj['y'])
            nx, ny = normals[idx]
            offset = min_offset + rng.uniform(0, spread_factor * disc_r)
            cx = bx + nx * offset
            cy = by + ny * offset
        else:
            angle = rng.uniform(0, 2 * np.pi)
            offset = obj['r'] + min_offset + rng.uniform(0, spread_factor * disc_r)
            cx = obj['cx'] + np.cos(angle) * offset
            cy = obj['cy'] + np.sin(angle) * offset
        if lo + margin <= cx <= hi - margin and lo + margin <= cy <= hi - margin:
            candidates.append((cx, cy))
    return candidates


# ── Packers ───────────────────────────────────────────────────────────────────

def pack_curves(
    n_curves=12,
    H=10,
    a=1.5,
    b=1.0,
    scale_range=(0.3, 1.0),
    domain=(-7, 7),
    gap=0.25,
    ds=0.1,
    max_attempts=3000,
    seed=None,
):
    """
    Pack non-overlapping random radial curves with minimum boundary gap `gap`.
    Each curve is discretized with arc-length spacing `ds`.
    """
    rng = np.random.default_rng(seed)
    lo_x, hi_x = domain
    lo_y, hi_y = -2, 2
    half_gap = gap / 2.0

    placed  = []
    results = []

    for i in range(n_curves):
        placed_ok = False
        for _ in range(max_attempts):
            curve_seed = int(rng.integers(0, 2**31))
            x, y = random_radial_curve(H=H, a=a, b=b, ds=ds, seed=curve_seed)

            s = rng.uniform(*scale_range)
            x *= s;  y *= s

            # Resample again after scaling so spacing stays = ds
            x, y = resample_by_arc_length(x, y, ds)

            half_w = (x.max() - x.min()) / 2.0
            half_h = (y.max() - y.min()) / 2.0
            margin = half_gap
            lo_cx, hi_cx = lo_x + half_w + margin, hi_x - half_w - margin
            lo_cy, hi_cy = lo_y + half_h + margin, hi_y - half_h - margin
            if lo_cx >= hi_cx or lo_cy >= hi_cy:
                continue

            cx = rng.uniform(lo_cx, hi_cx)
            cy = rng.uniform(lo_cy, hi_cy)
            xp, yp = x + cx, y + cy

            xe, ye   = expand_polygon(xp, yp, half_gap)
            xy       = np.column_stack([xp, yp])
            xy_exp   = np.column_stack([xe, ye])
            bbox_exp = curve_bbox(xe, ye)

            collision = False
            for (pl_xy, pl_xy_exp, pl_bbox_exp) in placed:
                if bboxes_overlap(bbox_exp, pl_bbox_exp, gap=0.0):
                    if polygons_collide(xy, xy_exp, pl_xy, pl_xy_exp):
                        collision = True
                        break

            if not collision:
                placed.append((xy, xy_exp, bbox_exp))
                results.append({
                    'type':    'curve',
                    'x':       xp,
                    'y':       yp,
                    'normals': compute_normals_2d(xp, yp),
                    'cx':      cx,
                    'cy':      cy,
                    'scale':   s,
                    'ds':      ds,
                })
                placed_ok = True
                break

        if not placed_ok:
            print(f"  Warning: could not place curve {i + 1} of {n_curves} "
                  f"after {max_attempts} attempts — skipping.")

    return results


def pack_discs(
    placed_objects,
    n_discs=20,
    disc_r=0.5,
    domain=(-7, 7),
    gap=0.25,
    ds=0.1,
    proximity_bias=0.75,
    max_attempts=2000,
    seed=None,
):
    """
    Pack non-overlapping discs of radius `disc_r`, biased to land near existing
    objects.  Each disc is also discretized with arc-length spacing `ds` (stored
    in 'x','y' for normal computation and plotting consistency).
    """
    rng = np.random.default_rng(seed)
    lo_x, hi_x = domain
    lo_y, hi_y = -2, 2
    margin = disc_r + gap / 2.0

    # Pre-build polygon data for already-placed curves
    curve_polys = []
    for obj in placed_objects:
        if obj['type'] == 'curve':
            xp, yp = obj['x'], obj['y']
            xe, ye = expand_polygon(xp, yp, gap / 2.0)
            curve_polys.append((
                np.column_stack([xp, yp]),
                np.column_stack([xe, ye]),
                curve_bbox(xe, ye),
            ))

    placed_discs = []
    results = []

    for i in range(n_discs):
        placed_ok = False
        for _ in range(max_attempts):
            use_proximity = (
                len(placed_objects) > 0 and rng.random() < proximity_bias
            )
            if use_proximity:
                candidates = sample_near_objects(
                    placed_objects, rng, domain, disc_r, gap,
                    n_candidates=8, spread_factor=2.0,
                )
                if candidates:
                    cx, cy = candidates[rng.integers(len(candidates))]
                else:
                    cx = rng.uniform(lo_x + margin, hi_x - margin)
                    cy = rng.uniform(lo_y + margin, hi_y - margin)
            else:
                cx = rng.uniform(lo_x + margin, hi_x - margin)
                cy = rng.uniform(lo_y + margin, hi_y - margin)

            if not (lo_x + margin <= cx <= hi_x - margin and
                    lo_y + margin <= cy <= hi_y - margin):
                continue

            disc_bbox = circle_bbox(cx, cy, disc_r + gap / 2.0)
            collision = False

            for (poly_xy, poly_xy_exp, poly_bbox_exp) in curve_polys:
                if bboxes_overlap(disc_bbox, poly_bbox_exp, gap=0.0):
                    if circle_polygon_collide(cx, cy, disc_r,
                                              poly_xy, poly_xy_exp, gap):
                        collision = True
                        break

            if not collision:
                for (dx, dy, dr) in placed_discs:
                    if circle_circle_collide(cx, cy, disc_r, dx, dy, dr, gap):
                        collision = True
                        break

            if not collision:
                # Discretize disc boundary at arc-length spacing ds
                xd, yd = circle_points(cx, cy, disc_r, ds)
                placed_discs.append((cx, cy, disc_r))
                disc_result = {
                    'type':    'disc',
                    'cx':      cx,
                    'cy':      cy,
                    'r':       disc_r,
                    'x':       xd,
                    'y':       yd,
                    'normals': compute_normals_2d(xd, yd),
                    'ds':      ds,
                }
                results.append(disc_result)
                placed_objects.append(disc_result)
                placed_ok = True
                break

        if not placed_ok:
            print(f"  Warning: could not place disc {i + 1} of {n_discs} "
                  f"after {max_attempts} attempts — skipping.")

    return results


# ── Verification helper ───────────────────────────────────────────────────────

def check_spacing(x, y, label=""):
    """Print min/mean/max chord length for a discretized closed curve."""
    xc = np.append(x, x[0])
    yc = np.append(y, y[0])
    chord = np.hypot(np.diff(xc), np.diff(yc))
    print(f"  {label:30s}  n={len(x):4d}  "
          f"ds min={chord.min():.4f}  mean={chord.mean():.4f}  max={chord.max():.4f}")

def make_chip_disc_chip(
    debye_length=0.1,
    H=6,
    a=0.5,
    b=1.0,
    scale=0.7,
    disc_r=0.1, 
    ds=0.06, #0.20, #0.25
    seed=42,
):
    rng = np.random.default_rng(seed)
    gap = debye_length

    # ── Generate and scale both chips ────────────────────────────────────────
    def make_chip(chip_seed):
        x, y = random_radial_curve(H=H, a=a, b=b, ds=ds, seed=chip_seed)
        x *= scale
        y *= scale
        x, y = resample_by_arc_length(x, y, ds)
        return x, y

    seed_b = int(rng.integers(0, 2**31))
    seed_t = int(rng.integers(0, 2**31))
    xb, yb = make_chip(seed_b)
    xt, yt = make_chip(seed_t)

    # ── Build assembly with bottom chip centred at origin (temporary) ─────────
    xb_p, yb_p = xb.copy(), yb.copy()

    top_of_bottom = yb_p.max()
    cy_d = top_of_bottom + gap + disc_r          # disc centre (before shift)
    cx_d = 0.0

    disc_top = cy_d + disc_r
    cy_t = disc_top + gap - yt.min()
    cx_t = 0.0
    xt_p, yt_p = xt + cx_t, yt + cy_t

    # ── Shift entire assembly so disc centre lands at (0, 0) ─────────────────
    shift = cy_d                                 # amount to subtract from all y
    yb_p -= shift
    yt_p -= shift
    cy_d  = 0.0
    cy_b  = -shift                               # centre of bottom chip
    cy_t -= shift

    xd, yd = circle_points(cx_d, cy_d, disc_r, ds)

    # ── Build return dicts ────────────────────────────────────────────────────
    chip_bottom = {
        'type':    'curve',
        'x':       xb_p,
        'y':       yb_p,
        'normals': compute_normals_2d(xb_p, yb_p),
        'cx':      cx_d,
        'cy':      cy_b,
        'scale':   scale,
        'ds':      ds,
    }
    disc = {
        'type':    'disc',
        'cx':      cx_d,
        'cy':      cy_d,
        'r':       disc_r,
        'x':       xd,
        'y':       yd,
        'normals': compute_normals_2d(xd, yd),
        'ds':      ds,
    }
    chip_top = {
        'type':    'curve',
        'x':       xt_p,
        'y':       yt_p,
        'normals': compute_normals_2d(xt_p, yt_p),
        'cx':      cx_t,
        'cy':      cy_t,
        'scale':   scale,
        'ds':      ds,
    }

    return chip_bottom, chip_top, disc

def get_c_d_c():
    chip_b, chip_t, d = make_chip_disc_chip()

    curves = []
    discs = []
    curves.append(chip_b)
    curves.append(chip_t)
    discs.append(d)
    
    xib = np.array([])
    yib = np.array([])
    n_x = np.array([])
    n_y = np.array([])

    for k, c in enumerate(curves):
        xib = np.concatenate([xib, c['x']])
        yib = np.concatenate([yib, c['y']])
        n_x = np.concatenate([n_x, c['normals'][:, 0]])
        n_y = np.concatenate([n_y, c['normals'][:, 1]])

    for d in discs:
        xib = np.concatenate([xib, d['x']])
        yib = np.concatenate([yib, d['y']])
        n_x = np.concatenate([n_x, d['normals'][:, 0]])
        n_y = np.concatenate([n_y, d['normals'][:, 1]])

    return xib, yib, n_x, n_y

def make_chip_suspension():
    curves_1, discs_1 = make_chip_suspension_left()
    curves_2, discs_2 = make_chip_suspension_right()

    curves = curves_1 + curves_2
    discs = discs_1 + discs_2

    return curves, discs

def get_chip_suspension():
    curves_1, discs_1 = make_chip_suspension_left()
    curves_2, discs_2 = make_chip_suspension_right()

    curves = curves_1 + curves_2
    discs = discs_1 + discs_2
    
    xib = np.array([])
    yib = np.array([])
    n_x = np.array([])
    n_y = np.array([])

    for k, c in enumerate(curves):
        xib = np.concatenate([xib, c['x']])
        yib = np.concatenate([yib, c['y']])
        n_x = np.concatenate([n_x, c['normals'][:, 0]])
        n_y = np.concatenate([n_y, c['normals'][:, 1]])

    for d in discs:
        xib = np.concatenate([xib, d['x']])
        yib = np.concatenate([yib, d['y']])
        n_x = np.concatenate([n_x, d['normals'][:, 0]])
        n_y = np.concatenate([n_y, d['normals'][:, 1]])

    return xib, yib, n_x, n_y

def make_chip_suspension_right():
    N_CURVES    = 8
    N_DISCS     = 8
    DISC_R      = 0.1
    N_HARMONICS = 10
    A, B        = 0.5, 1.0
    SCALE_RANGE = (0.25, 1.1)
    DOMAIN      = (0.5, 3.5)
    GAP         = 0.1
    DS          = 0.06#0.08    # ← target arc-length spacing for all objects
    SEED        = 42

    print(f"Packing {N_CURVES} curves (ds={DS}) …")
    curves = pack_curves(
        n_curves=N_CURVES,
        H=N_HARMONICS,
        a=A, b=B,
        scale_range=SCALE_RANGE,
        domain=DOMAIN,
        gap=GAP,
        ds=DS,
        max_attempts=3000,
        seed=SEED,
    )
    print(f"  Placed {len(curves)} / {N_CURVES} curves.")

    all_objects = list(curves)

    print(f"Packing {N_DISCS} discs (r={DISC_R}, ds={DS}) …")
    discs = pack_discs(
        placed_objects=all_objects,
        n_discs=N_DISCS,
        disc_r=DISC_R,
        domain=DOMAIN,
        gap=GAP,
        ds=DS,
        proximity_bias=0.75,
        max_attempts=2000,
        seed=SEED + 1 if SEED is not None else None,
    )
    print(f"  Placed {len(discs)} / {N_DISCS} discs.")

    return curves, discs 

def make_chip_suspension_left():
    N_CURVES    = 8
    N_DISCS     = 8
    DISC_R      = 0.1
    N_HARMONICS = 10
    A, B        = 0.5, 1.0
    SCALE_RANGE = (0.25, 1.1)
    DOMAIN      = (-3.5, -0.5)
    GAP         = 0.1
    DS          = 0.06#0.08    # ← target arc-length spacing for all objects
    SEED        = 42

    print(f"Packing {N_CURVES} curves (ds={DS}) …")
    curves = pack_curves(
        n_curves=N_CURVES,
        H=N_HARMONICS,
        a=A, b=B,
        scale_range=SCALE_RANGE,
        domain=DOMAIN,
        gap=GAP,
        ds=DS,
        max_attempts=3000,
        seed=SEED,
    )
    print(f"  Placed {len(curves)} / {N_CURVES} curves.")

    all_objects = list(curves)

    print(f"Packing {N_DISCS} discs (r={DISC_R}, ds={DS}) …")
    discs = pack_discs(
        placed_objects=all_objects,
        n_discs=N_DISCS,
        disc_r=DISC_R,
        domain=DOMAIN,
        gap=GAP,
        ds=DS,
        proximity_bias=0.75,
        max_attempts=2000,
        seed=SEED + 1 if SEED is not None else None,
    )
    print(f"  Placed {len(discs)} / {N_DISCS} discs.")

    return curves, discs 

def get_pack():
    N_CURVES    = 40
    N_DISCS     = 40
    DISC_R      = 0.25
    N_HARMONICS = 10
    A, B        = 1.0, 2.0
    SCALE_RANGE = (0.25, 1.1)
    DOMAIN      = (-2*np.pi, 2*np.pi)
    GAP         = 0.25
    DS          = 0.08    # ← target arc-length spacing for all objects
    SEED        = 42

    print(f"Packing {N_CURVES} curves (ds={DS}) …")
    curves = pack_curves(
        n_curves=N_CURVES,
        H=N_HARMONICS,
        a=A, b=B,
        scale_range=SCALE_RANGE,
        domain=DOMAIN,
        gap=GAP,
        ds=DS,
        max_attempts=3000,
        seed=SEED,
    )
    print(f"  Placed {len(curves)} / {N_CURVES} curves.")

    all_objects = list(curves)

    print(f"Packing {N_DISCS} discs (r={DISC_R}, ds={DS}) …")
    discs = pack_discs(
        placed_objects=all_objects,
        n_discs=N_DISCS,
        disc_r=DISC_R,
        domain=DOMAIN,
        gap=GAP,
        ds=DS,
        proximity_bias=0.75,
        max_attempts=2000,
        seed=SEED + 1 if SEED is not None else None,
    )
    print(f"  Placed {len(discs)} / {N_DISCS} discs.")

    xib = np.array([])
    yib = np.array([])
    n_x = np.array([])
    n_y = np.array([])

    for k, c in enumerate(curves):
        xib = np.concatenate([xib, c['x']])
        yib = np.concatenate([yib, c['y']])
        n_x = np.concatenate([n_x, c['normals'][:, 0]])
        n_y = np.concatenate([n_y, c['normals'][:, 1]])

    for d in discs:
        xib = np.concatenate([xib, d['x']])
        yib = np.concatenate([yib, d['y']])
        n_x = np.concatenate([n_x, d['normals'][:, 0]])
        n_y = np.concatenate([n_y, d['normals'][:, 1]])

    return xib, yib, n_x, n_y


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    N_CURVES    = 15
    N_DISCS     = 15
    DISC_R      = 0.1
    N_HARMONICS = 10
    A, B        = 0.5, 1.0
    SCALE_RANGE = (0.25, 1.1)
    DOMAIN      = (-4, 4)
    GAP         = 0.1
    DS          = 0.12    # ← target arc-length spacing for all objects
    SEED        = 42
    
    curves_1, discs_1 = make_chip_suspension_left()
    curves_2, discs_2 = make_chip_suspension_right()

    curves = curves_1 + curves_2
    discs = discs_1 + discs_2

    # Verify spacing
    print("\nSpacing verification:")
    for k, c in enumerate(curves):
        check_spacing(c['x'], c['y'], f"curve {k + 1}")
    for k, d in enumerate(discs):
        check_spacing(d['x'], d['y'], f"disc  {k + 1}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10))
    curve_cmap = plt.get_cmap('tab20', len(curves))
    disc_color = '#4477AA'

    for k, c in enumerate(curves):
        color = curve_cmap(k)
        ax.fill(c['x'], c['y'], alpha=0.18, color=color)
        ax.plot(c['x'], c['y'], 'o-', color=color, linewidth=1.2,
                markersize=2.5, markerfacecolor=color)
        step = max(1, len(c['x']) // 20)
        ax.quiver(
            c['x'][::step], c['y'][::step],
            c['normals'][::step, 0], c['normals'][::step, 1],
            color=color, scale=30, width=0.003, alpha=0.55,
        )

    for d in discs:
        # Draw filled disc
        patch = mpatches.Circle(
            (d['cx'], d['cy']), d['r'],
            facecolor=disc_color, edgecolor='#1a3a6b',
            alpha=0.40, linewidth=1.2, zorder=2,
        )
        ax.add_patch(patch)
        # Show discretization points on disc boundary
        ax.plot(d['x'], d['y'], 'o', color='#1a3a6b',
                markersize=2.0, alpha=0.7, zorder=3)
        step = max(1, len(d['x']) // 12)
        ax.quiver(
            d['x'][::step], d['y'][::step],
            d['normals'][::step, 0], d['normals'][::step, 1],
            color='#1a3a6b', scale=30, width=0.003, alpha=0.50, zorder=3,
        )

    curve_patch = mpatches.Patch(facecolor='grey',      alpha=0.4,
                                  label=f'curves (n={len(curves)})')
    disc_patch  = mpatches.Patch(facecolor=disc_color,  alpha=0.4,
                                  label=f'discs r={DISC_R} (n={len(discs)})')
    ax.legend(handles=[curve_patch, disc_patch], loc='upper right', fontsize=9)

    lo, hi = DOMAIN
    ax.set_xlim(lo, hi);  ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('x');   ax.set_ylabel('y')
    ax.set_title(f'Packed curves + discs  (gap={GAP}, ds={DS}, seed={SEED})')
    plt.tight_layout()
    #plt.savefig('./random_curve_pack.png', dpi=150)
    plt.show()
    print("Done.")
    # N_CURVES    = 60
    # N_DISCS     = 60
    # DISC_R      = 0.25
    # N_HARMONICS = 10
    # A, B        = 1.0, 2.0
    # SCALE_RANGE = (0.25, 1.1)
    # DOMAIN      = (-8, 8)
    # GAP         = 0.25
    # DS          = 0.12    # ← target arc-length spacing for all objects
    # SEED        = 42

    # chip_b, chip_t, d = make_chip_disc_chip()

    # curves = []
    # discs = []
    # curves.append(chip_b)
    # curves.append(chip_t)
    # discs.append(d)

    # # # Verify spacing
    # # print("\nSpacing verification:")
    # # for k, c in enumerate(curves):
    # #     check_spacing(c['x'], c['y'], f"curve {k + 1}")
    # # for k, d in enumerate(discs):
    # #     check_spacing(d['x'], d['y'], f"disc  {k + 1}")

    # # ── Plot ─────────────────────────────────────────────────────────────────
    # fig, ax = plt.subplots(figsize=(10, 10))
    # # curve_cmap = plt.get_cmap('tab20', len(curves))
    # disc_color = '#4477AA'

    # # for k, c in enumerate(curves):
    #     # color = curve_cmap(k)
    # ax.fill(chip_b['x'], chip_b['y'], alpha=0.18)
    # ax.plot(chip_b['x'], chip_b['y'], 'o-', linewidth=1.2,
    #         markersize=2.5)
    # step = max(1, len(chip_b['x']) // 20)
    # ax.quiver(
    #     chip_b['x'][::step], chip_b['y'][::step],
    #     chip_b['normals'][::step, 0], chip_b['normals'][::step, 1],
    #     scale=30, width=0.003, alpha=0.55,
    #     )
    
    # ax.fill(chip_t['x'], chip_t['y'], alpha=0.18)
    # ax.plot(chip_t['x'], chip_t['y'], 'o-', linewidth=1.2,
    #         markersize=2.5)
    # step = max(1, len(chip_t['x']) // 20)
    # ax.quiver(
    #     chip_t['x'][::step], chip_t['y'][::step],
    #     chip_t['normals'][::step, 0], chip_t['normals'][::step, 1],
    #     scale=30, width=0.003, alpha=0.55,
    #     )

    # # Draw filled disc
    # patch = mpatches.Circle(
    #     (d['cx'], d['cy']), d['r'],
    #     facecolor=disc_color, edgecolor='#1a3a6b',
    #     alpha=0.40, linewidth=1.2, zorder=2,
    # )
    # ax.add_patch(patch)
    # # Show discretization points on disc boundary
    # ax.plot(d['x'], d['y'], 'o', color='#1a3a6b',
    #         markersize=2.0, alpha=0.7, zorder=3)
    # step = max(1, len(d['x']) // 12)
    # ax.quiver(
    #     d['x'][::step], d['y'][::step],
    #     d['normals'][::step, 0], d['normals'][::step, 1],
    #     color='#1a3a6b', scale=30, width=0.003, alpha=0.50, zorder=3,
    # )

    # curve_patch = mpatches.Patch(facecolor='grey',      alpha=0.4,
    #                               label=f'curves (n={len(curves)})')
    # disc_patch  = mpatches.Patch(facecolor=disc_color,  alpha=0.4,
    #                               label=f'discs r={DISC_R} (n={len(discs)})')
    # ax.legend(handles=[curve_patch, disc_patch], loc='upper right', fontsize=9)

    # lo, hi = DOMAIN
    # ax.set_xlim(lo, hi);  ax.set_ylim(lo, hi)
    # ax.set_aspect('equal')
    # ax.set_xlabel('x');   ax.set_ylabel('y')
    # ax.set_title(f'Packed curves + discs  (gap={GAP}, ds={DS}, seed={SEED})')
    # plt.tight_layout()
    # #plt.savefig('./random_curve_pack.png', dpi=150)
    # plt.show()
    # print("Done.")