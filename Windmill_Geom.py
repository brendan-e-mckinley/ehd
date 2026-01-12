import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from sklearn.neighbors import NearestNeighbors
import networkx as netx

def order_curve_mst(x, y, k=8):
    pts = np.column_stack((x, y))
    N = len(pts)

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(pts)
    dists, inds = nbrs.kneighbors(pts)

    G = netx.Graph()
    for i in range(N):
        for j, d in zip(inds[i,1:], dists[i,1:]):
            G.add_edge(i, j, weight=d)

    # Minimum spanning tree
    T = netx.minimum_spanning_tree(G)

    # Find endpoints (degree 1 nodes)
    endpoints = [n for n in T.nodes if T.degree[n] == 1]

    # Longest path between endpoints = curve order
    max_len = 0
    best_path = None
    for i in endpoints:
        lengths, paths = netx.single_source_dijkstra(T, i)
        for j in endpoints:
            if lengths[j] > max_len:
                max_len = lengths[j]
                best_path = paths[j]

    return np.array(best_path)

# ------------------------------------------------------
# spline helpers (MINIMAL)
# ------------------------------------------------------
def bezier(P0, P1, P2, P3, t):
    return ((1-t)**3)*P0 + 3*(1-t)**2*t*P1 + 3*(1-t)*t**2*P2 + t**3*P3

def spline_between(cA, cB, R, dx, alpha=1.3):
    # direction between centers
    d = cB - cA
    L = np.linalg.norm(d)
    d /= L

    # attachment points (radial)
    P0 = cA + R*d
    P3 = cB - R*d

    # outward normal for this arm
    n = np.array([-d[1], d[0]])

    # tangents (circle tangents)
    T0 = n
    T3 = n

    # control points
    P1 = P0 + alpha*R*T0
    P2 = P3 + alpha*R*T3

    # discretization
    N = max(5, int(np.ceil(np.linalg.norm(P3-P0)/dx)))
    t = np.linspace(0, 1, N)

    C = ((1-t)**3)[:,None]*P0 \
      + 3*((1-t)**2*t)[:,None]*P1 \
      + 3*((1-t)*t**2)[:,None]*P2 \
      + (t**3)[:,None]*P3

    # normals
    dC = np.gradient(C, axis=0)
    tC = dC / np.linalg.norm(dC, axis=1)[:,None]
    nC = np.column_stack([tC[:,1], -tC[:,0]])

    return C[:,0], C[:,1], nC[:,0], nC[:,1]

# ------------------------------------------------------
# parameters
# ------------------------------------------------------
dx = 0.02
R  = 0.25
shift = 0.05

# ------------------------------------------------------
# triangle layout (with adjustable spacing)
# ------------------------------------------------------
sep = 1.4      # spacing factor
h = np.sqrt(3)*R

centers = [
    sep*np.array([-R - shift, -h/2]),   # bottom-left
    sep*np.array([ R + shift, -h/2]),   # bottom-right
    sep*np.array([ 0.0,  h/2 + shift])  # top-middle
]

# ------------------------------------------------------
# discretization
# ------------------------------------------------------
dth = dx / R
theta = np.arange(0, 2*np.pi, dth)

# ------------------------------------------------------
# build circles
# ------------------------------------------------------
xib = []
yib = []
n_x = []
n_y = []

i = 0

for c in centers:
    if (i == 0):
        lower_bound = np.pi / 4 - 3 * np.pi / 12
        upper_bound = np.pi / 4 + np.pi / 12
        mask = (theta < lower_bound) | (theta > upper_bound)
        
        masked_theta = theta[mask]
        
        x = c[0] + R*np.cos(masked_theta)
        y = c[1] + R*np.sin(masked_theta)

        nx = np.cos(masked_theta)
        ny = np.sin(masked_theta)

        xib.append(x)
        yib.append(y)
        n_x.append(nx)
        n_y.append(ny)

        xib_concat = np.concatenate(xib)
        yib_concat = np.concatenate(yib)
        n_x_concat = np.concatenate(n_x)
        n_y_concat = np.concatenate(n_y)

        xs, ys, nxs, nys = spline_between(
            centers[2], centers[0], R, dx, alpha=0.3
        )
        xib_concat = np.concatenate([xs, xib_concat])
        yib_concat = np.concatenate([ys, yib_concat])
        n_x_concat = np.concatenate([nxs, n_x_concat])
        n_y_concat = np.concatenate([nys, n_y_concat])

        plt.figure(figsize=(5,5))
        plt.scatter(xib_concat, yib_concat, s=6)
        plt.axis("equal")
        plt.title("Three disjoint IB circles")
        plt.show()
    if (i == 1):
        lower_bound = 3 * np.pi / 4 - np.pi / 12
        upper_bound = 3 * np.pi / 4 + 3 * np.pi / 12

        mask = (theta < lower_bound) | (theta > upper_bound)
        
        masked_theta = theta[mask]
        
        x = c[0] + R*np.cos(masked_theta)
        y = c[1] + R*np.sin(masked_theta)

        nx = np.cos(masked_theta)
        ny = np.sin(masked_theta)

        xib.append(x)
        yib.append(y)
        n_x.append(nx)
        n_y.append(ny)

        xib_concat = np.concatenate([xib_concat, np.concatenate(xib)])
        yib_concat = np.concatenate([yib_concat, np.concatenate(yib)])
        n_x_concat = np.concatenate([n_x_concat, np.concatenate(n_x)])
        n_y_concat = np.concatenate([n_y_concat, np.concatenate(n_y)])

        xs, ys, nxs, nys = spline_between(
            centers[0], centers[1], R, dx, alpha=0.3
        )
        xib_concat = np.concatenate([xib_concat, xs])
        yib_concat = np.concatenate([yib_concat, ys])
        n_x_concat = np.concatenate([n_x_concat, nxs])
        n_y_concat = np.concatenate([n_y_concat, nys])

        plt.figure(figsize=(5,5))
        plt.scatter(xib_concat, yib_concat, s=6)
        plt.axis("equal")
        plt.title("Three disjoint IB circles")
        plt.show()
    if (i == 2):
        lower_bound = 3 * np.pi / 2 - 2 * np.pi / 12
        upper_bound = 3 * np.pi / 2 + 2 * np.pi / 12

        mask = (theta < lower_bound) | (theta > upper_bound)
        
        masked_theta = theta[mask]
        
        x = c[0] + R*np.cos(masked_theta)
        y = c[1] + R*np.sin(masked_theta)

        nx = np.cos(masked_theta)
        ny = np.sin(masked_theta)

        xib.append(x)
        yib.append(y)
        n_x.append(nx)
        n_y.append(ny)

        xib_concat = np.concatenate([xib_concat, np.concatenate(xib)])
        yib_concat = np.concatenate([yib_concat, np.concatenate(yib)])
        n_x_concat = np.concatenate([n_x_concat, np.concatenate(n_x)])
        n_y_concat = np.concatenate([n_y_concat, np.concatenate(n_y)])

        xs, ys, nxs, nys = spline_between(
            centers[1], centers[2], R, dx, alpha=0.3
        )
        xib_concat = np.concatenate([xib_concat, xs])
        yib_concat = np.concatenate([yib_concat, ys])
        n_x_concat = np.concatenate([n_x_concat, nxs])
        n_y_concat = np.concatenate([n_y_concat, nys])

        plt.figure(figsize=(5,5))
        plt.scatter(xib_concat, yib_concat, s=6)
        plt.axis("equal")
        plt.title("Three disjoint IB circles")
        plt.show()

    i +=1 

# ------------------------------------------------------
# add three spline connectors
# ------------------------------------------------------

# pairs = [
#     (2, 0),  # top → left
#     (0, 1),  # left → right
#     (1, 2),  # right → top
# ]

# for i, j in pairs:
#     xs, ys, nxs, nys = spline_between(
#         centers[i], centers[j], R, dx, alpha=0.3
#     )
#     xib = np.concatenate([xib, xs])
#     yib = np.concatenate([yib, ys])
#     n_x = np.concatenate([n_x, nxs])
#     n_y = np.concatenate([n_y, nys])

# for col_index in range(nodes.shape[1] - 1):
#     xs, ys, nxs, nys = spline_between(
#         nodes[:, col_index], nodes[:, col_index + 1], R, dx, alpha=0.1
#     )
#     xib = np.concatenate([xib, xs])
#     yib = np.concatenate([yib, ys])
#     n_x = np.concatenate([n_x, nxs])
#     n_y = np.concatenate([n_y, nys])

Nib = len(xib_concat)

# for idx, value in enumerate(xib_concat):
#     plt.scatter(xib_concat[0:idx], yib_concat[0:idx])
#     plt.show()

# ------------------------------------------------------
# quick plot (sanity check)
# ------------------------------------------------------
plt.figure(figsize=(5,5))
plt.scatter(xib_concat, yib_concat, s=6)
plt.axis("equal")
plt.title("Three disjoint IB circles")
plt.show()

order = order_curve_mst(xib_concat, yib_concat)

xib_ord = xib_concat[order]
yib_ord = yib_concat[order]
n_x_ord = n_x_concat[order]
n_y_ord = n_y_concat[order]

# After ordering
x = xib_ord
y = yib_ord

# Compute distances between consecutive points
ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)

# Keep points that move by more than a tolerance
tol = 1e-10
keep = np.hstack(([True], ds > tol))

xib_ord = x[keep]
yib_ord = y[keep]
n_x_ord = n_x_ord[keep]
n_y_ord = n_y_ord[keep]

plt.figure(figsize=(5,5))
plt.plot(xib_ord, yib_ord, '-o', ms=2)
plt.axis('equal')
plt.title("Ordered IB curve")
plt.show()

# for idx, value in enumerate(xib_ord):
#     plt.scatter(xib_ord[0:idx], yib_ord[0:idx])
#     plt.show()

# ------------------------------------------------------
# save
# ------------------------------------------------------
savemat("Windmill_Geom.mat", {
    "xib": xib_ord,
    "yib": yib_ord,
    "n_x": n_x_ord,
    "n_y": n_y_ord
})