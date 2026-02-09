import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import amrex_poisson_3d

def test_poisson_3d():
    """Test 3D Poisson solver with known analytical solution."""
    
    # Initialize AMReX
    amrex_poisson_3d.amrex_init([])
    
    try:
        # Grid parameters
        nx, ny, nz = 64, 64, 64
        x_lo, x_hi = 0.0, 2.0 * np.pi
        y_lo, y_hi = 0.0, 2.0 * np.pi
        z_lo, z_hi = 0.0, 2.0 * np.pi
        
        # Create coordinate arrays
        x = np.linspace(x_lo, x_hi, nx, endpoint=False)
        y = np.linspace(y_lo, y_hi, ny, endpoint=False)
        z = np.linspace(z_lo, z_hi, nz, endpoint=False)
        
        # Create 3D meshgrid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Analytical solution: phi = sin(x) * sin(y) * sin(z)
        phi_exact = np.sin(X) * np.sin(Y) * np.sin(Z)
        
        # RHS from Laplacian: ∇²phi = -3*sin(x)*sin(y)*sin(z)
        # Note: AMReX solves -∇²phi = rhs, so we need rhs = 3*sin(x)*sin(y)*sin(z)
        rhs = 3.0 * np.sin(X) * np.sin(Y) * np.sin(Z)
        
        print(f"Grid size: {nx} x {ny} x {nz}")
        print(f"RHS shape: {rhs.shape}")
        print(f"RHS min/max: {rhs.min():.6f} / {rhs.max():.6f}")
        
        # Solve Poisson equation
        phi_numerical = amrex_poisson_3d.solve_poisson(
            rhs,
            x_lo=x_lo, x_hi=x_hi,
            y_lo=y_lo, y_hi=y_hi,
            z_lo=z_lo, z_hi=z_hi,
            tol=1e-10,
            nghost=1
        )
        
        print(f"Solution shape: {phi_numerical.shape}")
        print(f"Solution min/max: {phi_numerical.min():.6f} / {phi_numerical.max():.6f}")
        
        # Compute error
        error = phi_numerical - phi_exact
        max_error = np.abs(error).max()
        rms_error = np.sqrt(np.mean(error**2))
        
        print(f"\nError Analysis:")
        print(f"Max error: {max_error:.6e}")
        print(f"RMS error: {rms_error:.6e}")
        
        # Plot results
        plot_results(X, Y, Z, phi_exact, phi_numerical, error)
        
    finally:
        # Clean up AMReX
        amrex_poisson_3d.amrex_finalize()

def plot_results(X, Y, Z, phi_exact, phi_numerical, error):
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
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].contourf(X[:, :, mid_z], Y[:, :, mid_z], phi_numerical[:, :, mid_z], levels=20, cmap='RdBu_r')
    axes[0, 1].set_title('Numerical Solution (z=π)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].contourf(X[:, :, mid_z], Y[:, :, mid_z], error[:, :, mid_z], levels=20, cmap='seismic')
    axes[0, 2].set_title('Error (z=π)')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Row 2: XZ slice at mid-y
    im4 = axes[1, 0].contourf(X[:, mid_y, :], Z[:, mid_y, :], phi_exact[:, mid_y, :], levels=20, cmap='RdBu_r')
    axes[1, 0].set_title('Exact Solution (y=π)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('z')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].contourf(X[:, mid_y, :], Z[:, mid_y, :], phi_numerical[:, mid_y, :], levels=20, cmap='RdBu_r')
    axes[1, 1].set_title('Numerical Solution (y=π)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('z')
    plt.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].contourf(X[:, mid_y, :], Z[:, mid_y, :], error[:, mid_y, :], levels=20, cmap='seismic')
    axes[1, 2].set_title('Error (y=π)')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('z')
    plt.colorbar(im6, ax=axes[1, 2])
    
    # Row 3: YZ slice at mid-x
    im7 = axes[2, 0].contourf(Y[mid_x, :, :], Z[mid_x, :, :], phi_exact[mid_x, :, :], levels=20, cmap='RdBu_r')
    axes[2, 0].set_title('Exact Solution (x=π)')
    axes[2, 0].set_xlabel('y')
    axes[2, 0].set_ylabel('z')
    plt.colorbar(im7, ax=axes[2, 0])
    
    im8 = axes[2, 1].contourf(Y[mid_x, :, :], Z[mid_x, :, :], phi_numerical[mid_x, :, :], levels=20, cmap='RdBu_r')
    axes[2, 1].set_title('Numerical Solution (x=π)')
    axes[2, 1].set_xlabel('y')
    axes[2, 1].set_ylabel('z')
    plt.colorbar(im8, ax=axes[2, 1])
    
    im9 = axes[2, 2].contourf(Y[mid_x, :, :], Z[mid_x, :, :], error[mid_x, :, :], levels=20, cmap='seismic')
    axes[2, 2].set_title('Error (x=π)')
    axes[2, 2].set_xlabel('y')
    axes[2, 2].set_ylabel('z')
    plt.colorbar(im9, ax=axes[2, 2])
    
    plt.tight_layout()
    plt.savefig('poisson_3d_results.png', dpi=150)
    print("\nPlot saved as 'poisson_3d_results.png'")
    plt.show()

def test_constant_rhs():
    """Simple test with constant RHS."""
    amrex_poisson_3d.amrex_init([])
    
    try:
        nx, ny, nz = 32, 32, 32
        rhs = np.ones((nx, ny, nz))
        
        phi = amrex_poisson_3d.solve_poisson(rhs, tol=1e-8)
        
        print(f"\nConstant RHS test:")
        print(f"RHS = 1.0 everywhere")
        print(f"Solution min/max: {phi.min():.6f} / {phi.max():.6f}")
        
    finally:
        amrex_poisson_3d.amrex_finalize()

if __name__ == "__main__":
    print("="*60)
    print("Testing 3D Poisson Solver with Analytical Solution")
    print("="*60)
    test_poisson_3d()
    
    print("\n" + "="*60)
    print("Testing with Constant RHS")
    print("="*60)
    test_constant_rhs()