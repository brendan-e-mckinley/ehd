import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags, kron, eye
from scipy.linalg import cho_factor, cho_solve
import amrex_poisson

# --- Parameters ---
Nx = 4
Ny = 4
x_lo, x_hi = 0.0, 1.0  # Physical domain same as FD
y_lo, y_hi = 0.0, 1.0
dx = (x_hi - x_lo) / (Nx + 1)
dy = (y_hi - y_lo) / (Ny + 1)

# --- Simple RHS for testing ---
rhs = np.ones((Ny, Nx))  # uniform source

# --- FD Laplacian setup ---
e_x = (1/dx**2) * np.ones(Nx)
e_y = (1/dy**2) * np.ones(Ny)

D2_x = spdiags([e_x, -2*e_x, e_x], [-1, 0, 1], Nx, Nx)
D2_x_lil = D2_x.tolil()
D2_x_lil[0,0] = -3/dx**2
D2_x_lil[-1,-1] = -3/dx**2
D2_x = D2_x_lil.todia()
D2_y = spdiags([e_y, -2*e_y, e_y], [-1, 0, 1], Ny, Ny)
D2_y_lil = D2_y.tolil()
D2_y_lil[0,0] = -3/dy**2
D2_y_lil[-1,-1] = -3/dy**2
D2_y = D2_y_lil.todia()
I_x = eye(Nx)
I_y = eye(Ny)
M = np.zeros((Ny, Nx))

# Corners
M[0,0]       = -4/(dx**2)
M[0,-1]      = -4/(dx**2)
M[-1,0]      = -4/(dx**2)
M[-1,-1]     = -4/(dx**2)

# Top and bottom edges (excluding corners)
M[0,1:-1]    = -2/(dx**2)
M[-1,1:-1]   = -2/(dx**2)

# Left and right edges (excluding corners)
M[1:-1,0]    = -2/(dx**2)
M[1:-1,-1]   = -2/(dx**2)
rhs_altered = rhs + M

Lap = -(kron(I_x, D2_y) + kron(D2_x, I_y))
#c, low = cho_factor(Lap.toarray())
#phi_fd_flat = cho_solve((c, low), rhs.ravel(order='F'))
phi_fd_flat = spsolve(Lap, rhs_altered.ravel(order='F'))
phi_fd = phi_fd_flat.reshape((Ny, Nx), order='F')

print("FD Cholesky solution:\n", phi_fd)

# --- AMReX solver wrapper ---
class AmrexDLap:
    def __init__(self, nx, ny, x_lo=0.0, x_hi=1.0, y_lo=0.0, y_hi=1.0,
                 tol=1e-10, nghost=1):
        self.nx = nx
        self.ny = ny
        self.x_lo = x_lo; self.x_hi = x_hi
        self.y_lo = y_lo; self.y_hi = y_hi
        self.tol = tol
        self.nghost = nghost
        amrex_poisson.amrex_init([])

    def solve_A(self, rhs_flat):
        phi2d =  rhs_flat.reshape((self.ny, self.nx), order='F')
        phi_out = amrex_poisson.solve_poisson(phi2d,
                                              x_lo=self.x_lo, x_hi=self.x_hi,
                                              y_lo=self.y_lo, y_hi=self.y_hi,
                                              tol=self.tol,
                                              nghost=0,
                                              fortran_order_rhs=True)
        # AMReX returns (ny,nx) C-order array; flip sign to match FD
        return (-phi_out).ravel(order='F')  

    def finalize(self):
        amrex_poisson.amrex_finalize()

# --- Run AMReX ---
dLap = AmrexDLap(Nx, Ny, x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi)
phi_amrex_flat = dLap.solve_A(rhs.ravel(order='F'))
phi_amrex = phi_amrex_flat.reshape((Ny, Nx), order='F')
dLap.finalize()

print("\nAMReX solution (sign flipped):\n", phi_amrex)

sol_amrex = Lap * phi_amrex.ravel(order='F')
sol = Lap * phi_fd.ravel(order='F')

print("\nDifference:\n", phi_fd - phi_amrex)
