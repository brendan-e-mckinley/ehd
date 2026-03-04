import sys
import importlib
import numpy as np

# Force clean import
if 'amrex_poisson_3d' in sys.modules:
    del sys.modules['amrex_poisson_3d']
importlib.invalidate_caches()

import amrex_poisson_3d

print("Testing simple case first...")
amrex_poisson_3d.amrex_init([])

try:
    # Very simple test case
    nx, ny, nz = 64, 64, 64
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 2*np.pi, ny)
    z = np.linspace(0, 2*np.pi, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Simple RHS
    rhs = -3.0 * np.sin(X) * np.sin(Y) * np.sin(Z)
    phi_exact = np.sin(X) * np.sin(Y) * np.sin(Z)
    
    print(f"\n1. Testing UNIFORM solver (baseline):")
    print(f"   Grid: {nx}³")
    phi_uniform = amrex_poisson_3d.solve_poisson(rhs, tol=1e-10)
    error_uniform = np.abs(phi_uniform - phi_exact).max()
    print(f"   Max error: {error_uniform:.6e}")
    print(f"   Solution range: [{phi_uniform.min():.6f}, {phi_uniform.max():.6f}]")
    print(f"   Has NaN: {np.isnan(phi_uniform).any()}")
    
    print(f"\n2. Testing AMR solver with SAME resolution:")
    print(f"   Coarse: 64³, Fine: none (ref_ratio=1, radius=0)")
    phi_amr = amrex_poisson_3d.solve_poisson_amr(
        rhs,
        tol=1e-10,
        coarse_nx=64,
        coarse_ny=64,
        coarse_nz=64,
        refine_radius=0.01,  # Very small radius
        ref_ratio=2  # Minimal refinement
    )
    error_amr = np.abs(phi_amr - phi_exact).max()
    print(f"   Max error: {error_amr:.6e}")
    print(f"   Solution range: [{phi_amr.min():.6f}, {phi_amr.max():.6f}]")
    print(f"   Has NaN: {np.isnan(phi_amr).any()}")
    
finally:
    amrex_poisson_3d.amrex_finalize()