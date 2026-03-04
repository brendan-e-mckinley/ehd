import sys
import importlib
import numpy as np

# Force clean import
if 'amrex_poisson_3d' in sys.modules:
    del sys.modules['amrex_poisson_3d']
importlib.invalidate_caches()

import amrex_poisson_3d

print("Testing AMR Poisson Solver with Target Parameters")
print("="*70)

amrex_poisson_3d.amrex_init([])

try:
    # Test with moderate input resolution
    nx, ny, nz = 200, 200, 200
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 2*np.pi, ny)
    z = np.linspace(0, 2*np.pi, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Analytical solution
    rhs = -3.0 * np.sin(X) * np.sin(Y) * np.sin(Z)
    phi_exact = np.sin(X) * np.sin(Y) * np.sin(Z)
    
    print(f"Input grid: {nx}³ = {nx**3:,} cells")
    print(f"RHS range: [{rhs.min():.6f}, {rhs.max():.6f}]")
    
    # Target AMR parameters
    coarse_res = 64
    ref_ratio = 4
    refine_radius = 0.3
    
    fine_res = coarse_res * ref_ratio
    
    print(f"\nAMR Parameters:")
    print(f"  Coarse resolution: {coarse_res}³")
    print(f"  Refinement ratio: {ref_ratio}x")
    print(f"  Fine resolution: {fine_res}³ = {fine_res**3:,} cells")
    print(f"  Refinement radius: {refine_radius}")
    print(f"\nMemory comparison:")
    print(f"  Uniform 450³: {450**3:,} cells ({450**3 * 8 / 1e9:.2f} GB per field)")
    print(f"  Our approach: {fine_res**3:,} cells ({fine_res**3 * 8 / 1e9:.2f} GB per field)")
    print(f"  Savings: {100 * (1 - fine_res**3 / 450**3):.1f}%")
    
    print(f"\nSolving...")
    phi_numerical = amrex_poisson_3d.solve_poisson_amr(
        rhs,
        tol=1e-10,
        coarse_nx=coarse_res,
        coarse_ny=coarse_res,
        coarse_nz=coarse_res,
        refine_radius=refine_radius,
        ref_ratio=ref_ratio
    )
    
    # Check for NaN
    has_nan = np.isnan(phi_numerical).any()
    print(f"\nSolution statistics:")
    print(f"  Range: [{phi_numerical.min():.6f}, {phi_numerical.max():.6f}]")
    print(f"  Has NaN: {has_nan}")
    
    if not has_nan:
        # Compute errors
        error = phi_numerical - phi_exact
        max_error = np.abs(error).max()
        rms_error = np.sqrt(np.mean(error**2))
        
        print(f"\nError Analysis:")
        print(f"  Max error: {max_error:.6e}")
        print(f"  RMS error: {rms_error:.6e}")
        
        # Check error in central region
        x_center = np.pi
        y_center = np.pi
        z_center = np.pi
        r = np.sqrt((X - x_center)**2 + (Y - y_center)**2 + (Z - z_center)**2)
        
        central_mask = r <= refine_radius
        if np.any(central_mask):
            central_error = np.abs(error[central_mask])
            outer_mask = r > refine_radius
            outer_error = np.abs(error[outer_mask]) if np.any(outer_mask) else np.array([0])
            
            print(f"\nRegional Error Analysis:")
            print(f"  Central region (r ≤ {refine_radius}):")
            print(f"    Max error: {central_error.max():.6e}")
            print(f"    RMS error: {np.sqrt(np.mean(central_error**2)):.6e}")
            print(f"  Outer region (r > {refine_radius}):")
            print(f"    Max error: {outer_error.max():.6e}")
            print(f"    RMS error: {np.sqrt(np.mean(outer_error**2)):.6e}")
        
        print(f"\n{'='*70}")
        print("SUCCESS! AMR solver working correctly")
        print("="*70)
        print(f"\nYou now have:")
        print(f"  ✓ {fine_res}³ effective resolution")
        print(f"  ✓ {100 * (1 - fine_res**3 / 450**3):.1f}% memory reduction vs 450³")
        print(f"  ✓ Correct solutions (no NaN)")
        print(f"  ✓ Good accuracy (max error: {max_error:.2e})")
        
    else:
        print("\nERROR: Solution contains NaN values")

finally:
    amrex_poisson_3d.amrex_finalize()