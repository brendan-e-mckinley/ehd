# ------------------------------------------------------
# Immersed boundary: snowman hull with smooth side connectors
# ------------------------------------------------------

# radii
R_big   = 0.50
R_small = 0.25

# centers
x_big,   y_big   = 0.0, 0.0
x_small, y_small = 0.0, R_big + 2*R_small

# angular spacings
dth_big   = dx / R_big
dth_small = dx / R_small

theta_big   = np.arange(0, 2*np.pi, dth_big)
theta_small = np.arange(0, 2*np.pi, dth_small)

# ------------------------------------------------------
# circles
# ------------------------------------------------------
xib_big = x_big + R_big * np.cos(theta_big)
yib_big = y_big + R_big * np.sin(theta_big)
n_x_big = np.cos(theta_big)
n_y_big = np.sin(theta_big)

xib_small = x_small + R_small * np.cos(theta_small)
yib_small = y_small + R_small * np.sin(theta_small)
n_x_small = np.cos(theta_small)
n_y_small = np.sin(theta_small)

# ------------------------------------------------------
# keep only exterior semicircles
# ------------------------------------------------------
# big circle: keep LOWER half only
keep_big_semicircle = yib_big <= y_big

# small circle: keep UPPER half only
keep_small_semicircle = yib_small >= y_small

# ------------------------------------------------------
# remove side arcs to make room for connectors
# ------------------------------------------------------
delta = np.pi / 6  # arc half-width

def side_mask(theta, side):
    if side == "right":
        return (theta < -delta) | (theta > delta)
    else:  # left
        return (theta < np.pi - delta) | (theta > np.pi + delta)

keep_big = keep_big_semicircle & \
           side_mask(theta_big, "right") & side_mask(theta_big, "left")

keep_small = keep_small_semicircle & \
             side_mask(theta_small, "right") & side_mask(theta_small, "left")

xib_big = xib_big[keep_big]
yib_big = yib_big[keep_big]
n_x_big = n_x_big[keep_big]
n_y_big = n_y_big[keep_big]

xib_small = xib_small[keep_small]
yib_small = yib_small[keep_small]
n_x_small = n_x_small[keep_small]
n_y_small = n_y_small[keep_small]

# ------------------------------------------------------
# cubic Bézier helpers
# ------------------------------------------------------
def bezier(P0, P1, P2, P3, t):
    return ((1-t)**3)*P0 + 3*(1-t)**2*t*P1 + 3*(1-t)*t**2*P2 + t**3*P3

def normals_from_curve(C):
    dC = np.gradient(C, axis=0)
    t = dC / np.linalg.norm(dC, axis=1)[:,None]
    n = np.column_stack([t[:,1], -t[:,0]])
    return n

# ------------------------------------------------------
# build smooth connectors
# ------------------------------------------------------
alpha = 1.5

# right connector
P0 = np.array([x_big + R_big, y_big])
P3 = np.array([x_small + R_small, y_small])

T0 = np.array([0.0,  R_big])
T3 = np.array([0.0, -R_small])

P1 = P0 + alpha*T0
P2 = P3 + alpha*T3

Nr = int(np.linalg.norm(P3 - P0) / dx)
t_vals = np.linspace(0, 1, Nr)
right_curve = np.array([bezier(P0, P1, P2, P3, t) for t in t_vals])
n_right = normals_from_curve(right_curve)

# left connector
P0 = np.array([x_small - R_small, y_small])
P3 = np.array([x_big - R_big, y_big])

T0 = np.array([0.0, -R_small])
T3 = np.array([0.0,  R_big])

P1 = P0 + alpha*T0
P2 = P3 + alpha*T3

left_curve = np.array([bezier(P0, P1, P2, P3, t) for t in t_vals])
n_left = normals_from_curve(left_curve)

# ------------------------------------------------------
# assemble boundary
# ------------------------------------------------------
xib = np.concatenate([
    xib_big,
    right_curve[:,0],
    xib_small,
    left_curve[:,0]
])

yib = np.concatenate([
    yib_big,
    right_curve[:,1],
    yib_small,
    left_curve[:,1]
])

n_x = np.concatenate([
    n_x_big,
    n_right[:,0],
    n_x_small,
    n_left[:,0]
])

n_y = np.concatenate([
    n_y_big,
    n_right[:,1],
    n_y_small,
    n_left[:,1]
])

Nib = len(xib)

# save snowman geometry
savemat('Snowman_Geom.mat', {
    'xib': xib,
    'yib': yib
})