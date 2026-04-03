import sys, importlib
if 'amrex_poisson_3d' in sys.modules:
    del sys.modules['amrex_poisson_3d']
importlib.invalidate_caches()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import amrex_poisson_3d

def apply_poisson(x_in, bcs, x_lo, x_hi,
                        y_lo, y_hi,
                        z_lo, z_hi):
    
    # Initialize AMReX
    amrex_poisson_3d.amrex_init([])

    lap_x = []
    
    try:
        # apply Laplacian operator
        lap_x = amrex_poisson_3d.apply_poisson(
            x_in,
            bcs,
            x_lo=x_lo, x_hi=x_hi,
            y_lo=y_lo, y_hi=y_hi,
            z_lo=z_lo, z_hi=z_hi,
            nghost=1,
            fortran_order_x=False,
            fortran_order_bc=False
        )
        
    finally:
        # Clean up AMReX
        amrex_poisson_3d.amrex_finalize()
        return lap_x

def solve_poisson(rhs, bcs, x_lo, x_hi,
                        y_lo, y_hi,
                        z_lo, z_hi, refine_radius):
    
    # Initialize AMReX
    amrex_poisson_3d.amrex_init([])

    phi_numerical = []
    
    try:
        # AMR grid parameters
        coarse_res = 64
        ref_ratio = 4
        
        print(f"\nAMR Grid Setup:")
        print(f"  Coarse grid: {coarse_res}³ = {coarse_res**3:,} cells")
        print(f"  Fine grid region: sphere of radius {refine_radius}")
        print(f"  Refinement ratio: {ref_ratio}x")
        print(f"  Effective resolution in center: {coarse_res*ref_ratio}³")
        
        # Solve Poisson equation with AMR
        phi_numerical = amrex_poisson_3d.solve_poisson_adaptive(
            rhs,
            bcs, 
            x_lo=x_lo, x_hi=x_hi,
            y_lo=y_lo, y_hi=y_hi,
            z_lo=z_lo, z_hi=z_hi,
            tol=1e-10,
            nghost=1,
            fortran_order_rhs=False,
            fortran_order_bc=False
        )
        
    finally:
        # Clean up AMReX
        amrex_poisson_3d.amrex_finalize()
        return phi_numerical

def plot_results(X, Y, Z, phi_exact, phi_numerical, error, refine_radius=None):
    """Visualize results with 2D slices through the 3D domain."""
    
    # Take middle slices
    nx, ny, nz = X.shape
    mid_x = nx // 2
    mid_y = ny // 2
    mid_z = nz // 2
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Row 1: XY slice at mid-z
    im1 = axes[0, 0].contourf(X[:, :, mid_z], Y[:, :, mid_z], phi_exact[:, :, mid_z], levels=20, cmap='RdBu_r')
    axes[0, 0].set_title('Exact Solution (z=π)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    if refine_radius:
        circle = plt.Circle((np.pi, np.pi), refine_radius, fill=False, edgecolor='green', linewidth=2, linestyle='--', label='AMR boundary')
        axes[0, 0].add_patch(circle)
        axes[0, 0].legend()
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].contourf(X[:, :, mid_z], Y[:, :, mid_z], phi_numerical[:, :, mid_z], levels=20, cmap='RdBu_r')
    axes[0, 1].set_title('Numerical Solution (z=π)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    if refine_radius:
        circle = plt.Circle((np.pi, np.pi), refine_radius, fill=False, edgecolor='green', linewidth=2, linestyle='--', label='AMR boundary')
        axes[0, 1].add_patch(circle)
        axes[0, 1].legend()
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].contourf(X[:, :, mid_z], Y[:, :, mid_z], error[:, :, mid_z], levels=20, cmap='seismic')
    axes[0, 2].set_title('Error (z=π)')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    if refine_radius:
        circle = plt.Circle((np.pi, np.pi), refine_radius, fill=False, edgecolor='green', linewidth=2, linestyle='--', label='AMR boundary')
        axes[0, 2].add_patch(circle)
        axes[0, 2].legend()
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Row 2: XZ slice at mid-y
    im4 = axes[1, 0].contourf(X[:, mid_y, :], Z[:, mid_y, :], phi_exact[:, mid_y, :], levels=20, cmap='RdBu_r')
    axes[1, 0].set_title('Exact Solution (y=π)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('z')
    if refine_radius:
        circle = plt.Circle((np.pi, np.pi), refine_radius, fill=False, edgecolor='green', linewidth=2, linestyle='--', label='AMR boundary')
        axes[1, 0].add_patch(circle)
        axes[1, 0].legend()
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].contourf(X[:, mid_y, :], Z[:, mid_y, :], phi_numerical[:, mid_y, :], levels=20, cmap='RdBu_r')
    axes[1, 1].set_title('Numerical Solution (y=π)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('z')
    if refine_radius:
        circle = plt.Circle((np.pi, np.pi), refine_radius, fill=False, edgecolor='green', linewidth=2, linestyle='--', label='AMR boundary')
        axes[1, 1].add_patch(circle)
        axes[1, 1].legend()
    plt.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].contourf(X[:, mid_y, :], Z[:, mid_y, :], error[:, mid_y, :], levels=20, cmap='seismic')
    axes[1, 2].set_title('Error (y=π)')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('z')
    if refine_radius:
        circle = plt.Circle((np.pi, np.pi), refine_radius, fill=False, edgecolor='green', linewidth=2, linestyle='--', label='AMR boundary')
        axes[1, 2].add_patch(circle)
        axes[1, 2].legend()
    plt.colorbar(im6, ax=axes[1, 2])
    
    # Row 3: YZ slice at mid-x
    im7 = axes[2, 0].contourf(Y[mid_x, :, :], Z[mid_x, :, :], phi_exact[mid_x, :, :], levels=20, cmap='RdBu_r')
    axes[2, 0].set_title('Exact Solution (x=π)')
    axes[2, 0].set_xlabel('y')
    axes[2, 0].set_ylabel('z')
    if refine_radius:
        circle = plt.Circle((np.pi, np.pi), refine_radius, fill=False, edgecolor='green', linewidth=2, linestyle='--', label='AMR boundary')
        axes[2, 0].add_patch(circle)
        axes[2, 0].legend()
    plt.colorbar(im7, ax=axes[2, 0])
    
    im8 = axes[2, 1].contourf(Y[mid_x, :, :], Z[mid_x, :, :], phi_numerical[mid_x, :, :], levels=20, cmap='RdBu_r')
    axes[2, 1].set_title('Numerical Solution (x=π)')
    axes[2, 1].set_xlabel('y')
    axes[2, 1].set_ylabel('z')
    if refine_radius:
        circle = plt.Circle((np.pi, np.pi), refine_radius, fill=False, edgecolor='green', linewidth=2, linestyle='--', label='AMR boundary')
        axes[2, 1].add_patch(circle)
        axes[2, 1].legend()
    plt.colorbar(im8, ax=axes[2, 1])
    
    im9 = axes[2, 2].contourf(Y[mid_x, :, :], Z[mid_x, :, :], error[mid_x, :, :], levels=20, cmap='seismic')
    axes[2, 2].set_title('Error (x=π)')
    axes[2, 2].set_xlabel('y')
    axes[2, 2].set_ylabel('z')
    if refine_radius:
        circle = plt.Circle((np.pi, np.pi), refine_radius, fill=False, edgecolor='green', linewidth=2, linestyle='--', label='AMR boundary')
        axes[2, 2].add_patch(circle)
        axes[2, 2].legend()
    plt.colorbar(im9, ax=axes[2, 2])
    
    plt.tight_layout()
    plt.savefig('poisson_3d_amr_results_ib.png', dpi=150)
    print("\nPlot saved as 'poisson_3d_amr_results.png'")
    plt.show()

if __name__ == "__main__":
    # Grid parameters for INPUT arrays
    # You can use whatever resolution you want for input - the solver will interpolate
    nx, ny, nz = 200, 200, 200  # Reduced from 450 for faster testing
    x_lo, x_hi = 0.0, 2 * np.pi
    y_lo, y_hi = 0.0, 2 * np.pi
    z_lo, z_hi = 0.0, 2 * np.pi

    refine_radius = 0.3
    
    # Create coordinate arrays
    x = np.linspace(x_lo, x_hi, nx)
    y = np.linspace(y_lo, y_hi, ny)
    z = np.linspace(z_lo, z_hi, nz)

    xint = x[1:-1]
    yint = y[1:-1]
    zint = z[1:-1]
    
    # Create 3D meshgrid
    Xint, Yint, Zint = np.meshgrid(xint, yint, zint, indexing='ij')
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Analytical solution: phi = sin(x) * sin(y) * sin(z)
    # This satisfies phi = 0 at all boundaries
    phi_exact = 20 + np.sin(X) * np.sin(Y) * np.sin(Z)
    
    # For -∇²phi = rhs, we have:
    # -∇²[sin(x)sin(y)sin(z)] = 3*sin(x)*sin(y)*sin(z)
    rhs = -3.0 * np.sin(X) * np.sin(Y) * np.sin(Z)
    boundary_values = np.zeros_like(X) + 20  # or whatever your outer BC is
    
    phi_numerical = solve_poisson(rhs, boundary_values, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, refine_radius)
    rhs_numerical = apply_poisson(phi_exact, boundary_values, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)

    #######################################
    #### CHECK ERROR FOR solve_poisson ####
    #######################################
    print(f"\nSolution shape: {phi_numerical.shape}")
    print(f"Solution min/max: {phi_numerical.min():.6f} / {phi_numerical.max():.6f}")
    
    # Compute error
    error = phi_numerical - phi_exact
    max_error = np.abs(error).max()
    rms_error = np.sqrt(np.mean(error**2))
    
    print(f"\nError Analysis:")
    print(f"Max error: {max_error:.6e}")
    print(f"RMS error: {rms_error:.6e}")
    
    # Compute error in central region (where we have fine grid)
    center_x = 0.5 * (x_lo + x_hi)
    center_y = 0.5 * (y_lo + y_hi)
    center_z = 0.5 * (z_lo + z_hi)
    
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2)
    central_mask = r <= refine_radius
    
    if np.any(central_mask):
        central_error = np.abs(error[central_mask])
        print(f"\nCentral region (r <= {refine_radius}) error:")
        print(f"  Max error: {central_error.max():.6e}")
        print(f"  RMS error: {np.sqrt(np.mean(central_error**2)):.6e}")
    
    # Plot results
    plot_results(X, Y, Z, phi_exact, phi_numerical, error, refine_radius)

    #######################################
    #### CHECK ERROR FOR apply_poisson ####
    #######################################
    rhs_numerical = rhs_numerical[1:-1,1:-1,1:-1]
    rhs = rhs[1:-1,1:-1,1:-1]

    print(f"\nSolution shape: {rhs_numerical.shape}")
    print(f"Solution min/max: {rhs_numerical.min():.6f} / {rhs_numerical.max():.6f}")
    
    # Compute error
    error = rhs_numerical - rhs
    max_error = np.abs(error).max()
    rms_error = np.sqrt(np.mean(error**2))
    
    print(f"\nError Analysis:")
    print(f"Max error: {max_error:.6e}")
    print(f"RMS error: {rms_error:.6e}")
    
    # Compute error in central region (where we have fine grid)
    center_x = 0.5 * (x_lo + x_hi)
    center_y = 0.5 * (y_lo + y_hi)
    center_z = 0.5 * (z_lo + z_hi)
    
    r = np.sqrt((X[1:-1,1:-1,1:-1] - center_x)**2 + (Y[1:-1,1:-1,1:-1] - center_y)**2 + (Z[1:-1,1:-1,1:-1] - center_z)**2)
    central_mask = r <= refine_radius
    
    if np.any(central_mask):
        central_error = np.abs(error[central_mask])
        print(f"\nCentral region (r <= {refine_radius}) error:")
        print(f"  Max error: {central_error.max():.6e}")
        print(f"  RMS error: {np.sqrt(np.mean(central_error**2)):.6e}")
    
    # Plot results
    plot_results(X[1:-1,1:-1,1:-1], Y[1:-1,1:-1,1:-1], Z[1:-1,1:-1,1:-1], rhs, rhs_numerical, error, refine_radius)