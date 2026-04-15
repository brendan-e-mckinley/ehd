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
    
def solve_poisson_double_grid(rhs_coarse, rhs_fine, bcs, x_lo, x_hi,
                                y_lo, y_hi,
                                z_lo, z_hi):
    
    # Initialize AMReX
    amrex_poisson_3d.amrex_init([])

    phi_numerical_coarse = []
    phi_numerical_fine = []
    
    try:
        # Solve Poisson equation with AMR
        phi_numerical_coarse, phi_numerical_fine = amrex_poisson_3d.solve_poisson_adaptive_double_grid(
            rhs_coarse,
            rhs_fine,
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
        return phi_numerical_coarse, phi_numerical_fine

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
    nx_coarse, ny_coarse, nz_coarse = 64, 64, 64
    ref_ratio = 2

    x_lo, x_hi = 0.0, 2 * np.pi
    y_lo, y_hi = 0.0, 2 * np.pi
    z_lo, z_hi = 0.0, 2 * np.pi

    # Coarse grid coordinates (cell-centered or node — match your solver convention)
    x_coarse = np.linspace(x_lo, x_hi, nx_coarse)
    y_coarse = np.linspace(y_lo, y_hi, ny_coarse)
    z_coarse = np.linspace(z_lo, z_hi, nz_coarse)

    X, Y, Z = np.meshgrid(x_coarse, y_coarse, z_coarse, indexing='ij')

    # coarse rhs
    rhs_coarse = -3.0 * np.sin(X) * np.sin(Y) * np.sin(Z)

    # FIX: fine patch covers the central 50% of the domain (matching cube_half_widths = 0.25*(hi-lo))
    # At ref_ratio=2, coarse has 64 cells, so the central half = 32 coarse cells = 64 fine cells
    nx_fine_patch = nx_coarse * ref_ratio // 2   # = 64
    ny_fine_patch = ny_coarse * ref_ratio // 2   # = 64
    nz_fine_patch = nz_coarse * ref_ratio // 2   # = 64

    # Fine patch physical extent: central 50% of the domain
    x_patch_lo = x_lo + 0.25 * (x_hi - x_lo)   # = pi/2
    x_patch_hi = x_hi - 0.25 * (x_hi - x_lo)   # = 3pi/2
    y_patch_lo = y_lo + 0.25 * (y_hi - y_lo)
    y_patch_hi = y_hi - 0.25 * (y_hi - y_lo)
    z_patch_lo = z_lo + 0.25 * (z_hi - z_lo)
    z_patch_hi = z_hi - 0.25 * (z_hi - z_lo)

    x_fine = np.linspace(x_patch_lo, x_patch_hi, nx_fine_patch)
    y_fine = np.linspace(y_patch_lo, y_patch_hi, ny_fine_patch)
    z_fine = np.linspace(z_patch_lo, z_patch_hi, nz_fine_patch)

    Xf, Yf, Zf = np.meshgrid(x_fine, y_fine, z_fine, indexing='ij')

    # FIX: rhs_fine is sized to the patch only, not the full refined domain
    rhs_fine = -3.0 * np.sin(Xf) * np.sin(Yf) * np.sin(Zf)

    boundary_values = np.zeros_like(X) + 20
    boundary_values_fine = np.zeros_like(Xf) + 20
    phi_exact = 20 + np.sin(X) * np.sin(Y) * np.sin(Z)
    phi_exact_fine = 20 + np.sin(Xf) * np.sin(Yf) * np.sin(Zf)

    phi_coarse, phi_fine = solve_poisson_double_grid(
        rhs_coarse, rhs_fine, boundary_values,
        x_lo, x_hi, y_lo, y_hi, z_lo, z_hi
    )

    print(f"Coarse solution shape: {phi_coarse.shape}")
    print(f"Fine solution shape:   {phi_fine.shape}")

    # Coarse error
    error_coarse = phi_coarse - phi_exact
    print(f"\nCoarse error — max: {np.abs(error_coarse).max():.6e}, "
          f"rms: {np.sqrt(np.mean(error_coarse**2)):.6e}")

    # Fine error (compare against exact on the patch)
    error_fine = phi_fine - phi_exact_fine
    print(f"Fine error   — max: {np.abs(error_fine).max():.6e}, "
          f"rms: {np.sqrt(np.mean(error_fine**2)):.6e}")
    
    # Plot results
    plot_results(X, Y, Z, phi_exact, phi_coarse, error_coarse, 0.3)
    plot_results(Xf, Yf, Zf, phi_exact_fine, phi_fine, error_fine, 0.3)