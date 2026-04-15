#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MLMG.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Array.H>
#include <AMReX_ParallelDescriptor.H>

#include <AMReX_TagBox.H>
#include <AMReX_FillPatchUtil.H>   // for InterpFromCoarseLevel
#include <AMReX_Interpolater.H>

namespace py = pybind11;
using namespace amrex;

auto rhs_function = [](Real x, Real y, Real z) -> Real {
    return -3.0 * std::sin(x) * std::sin(y) * std::sin(z);
};

// Helper: require 3D arrays for this version
static void check_3d_array(const py::array_t<double>& arr) {
    if (arr.ndim() != 3) {
        throw std::runtime_error("Only 3D arrays supported in this version.");
    }
}

py::object amrex_init(py::list py_argv = py::list()) {
    // Allow the caller to pass CLI args if desired. Default: no args.
    std::vector<const char*> argv;
    std::vector<std::string> argv_storage;
    argv_storage.reserve(py_argv.size());
    for (auto item : py_argv) {
        std::string s = py::cast<std::string>(item);
        argv_storage.push_back(s);
    }
    for (auto &s : argv_storage) argv.push_back(s.c_str());

    // AMReX provides overloads; to be robust call Initialize with argc/argv if present.
    int argc = static_cast<int>(argv.size());
    char** cargv = (argc>0) ? const_cast<char**>(argv.data()) : nullptr;
    amrex::Initialize(argc, cargv);
    return py::none();
}

py::object amrex_finalize() {
    amrex::Finalize();
    return py::none();
}

BoxArray make_refined_ba_cube(const Geometry& coarse_geom, const Box& coarse_domain,
    const Real cube_center[3], const Real cube_half_widths[3],
    int max_grid_size = 16)
{
    IntVect ref_ratio(AMREX_D_DECL(2,2,2));
    BoxArray coarse_ba(coarse_domain);
    coarse_ba.maxSize(max_grid_size);
    DistributionMapping coarse_dm(coarse_ba);
    const Real* dx      = coarse_geom.CellSize();
    const Real* prob_lo = coarse_geom.ProbLo();

    iMultiFab mask(coarse_ba, coarse_dm, 1, 0);
    mask.setVal(0);
    for (MFIter mfi(mask); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mask.array(mfi);
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
            Real x = prob_lo[0] + (i + 0.5) * dx[0];
            Real y = prob_lo[1] + (j + 0.5) * dx[1];
            Real z = prob_lo[2] + (k + 0.5) * dx[2];
            if (std::abs(x - cube_center[0]) < cube_half_widths[0] &&
                std::abs(y - cube_center[1]) < cube_half_widths[1] &&
                std::abs(z - cube_center[2]) < cube_half_widths[2])
                arr(i,j,k) = 1;
        }
    }

    BoxList bl;
    for (MFIter mfi(mask); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mask.const_array(mfi);
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
            if (arr(i,j,k) == 1) {
                Box coarse_cell(IntVect(i,j,k), IntVect(i,j,k));
                bl.push_back(coarse_cell.refine(ref_ratio));
            }
        }
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!bl.isEmpty(),
        "Refined BoxList is empty — cube may not overlap domain");
    bl.simplify();
    bl.maxSize(max_grid_size);
    return BoxArray(bl);
}

BoxArray make_refined_ba(const Geometry& coarse_geom, const Box& coarse_domain,
    const Real sphere_center[3], Real sphere_radius,
    int max_grid_size = 16)
{
    IntVect ref_ratio(AMREX_D_DECL(2,2,2));

    BoxArray coarse_ba(coarse_domain);
    coarse_ba.maxSize(max_grid_size);
    DistributionMapping coarse_dm(coarse_ba);

    const Real* dx      = coarse_geom.CellSize();
    const Real* prob_lo = coarse_geom.ProbLo();

    // Build a mask: 1 where we want refinement, 0 elsewhere
    iMultiFab mask(coarse_ba, coarse_dm, 1, 0);
    mask.setVal(0);

    for (MFIter mfi(mask); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mask.array(mfi);
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
            Real x = prob_lo[0] + (i + 0.5) * dx[0];
            Real y = prob_lo[1] + (j + 0.5) * dx[1];
            Real z = prob_lo[2] + (k + 0.5) * dx[2];
            Real d2 = (x-sphere_center[0])*(x-sphere_center[0])
                    + (y-sphere_center[1])*(y-sphere_center[1])
                    + (z-sphere_center[2])*(z-sphere_center[2]);
            if (d2 < sphere_radius * sphere_radius)
                arr(i,j,k) = 1;
        }
    }

    // Collect tagged coarse boxes into a BoxList, then refine to fine index space
    BoxList bl;
    for (MFIter mfi(mask); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mask.const_array(mfi);
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
            if (arr(i,j,k) == 1) {
                // Add this single coarse cell as a box, then refine it
                Box coarse_cell(IntVect(i,j,k), IntVect(i,j,k));
                bl.push_back(coarse_cell.refine(ref_ratio));
            }
        }
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!bl.isEmpty(),
        "Refined BoxList is empty — sphere may not overlap domain");

    bl.simplify();           // merge adjacent boxes where possible
    bl.maxSize(max_grid_size);
    return BoxArray(bl);
}

py::array_t<double> solve_poisson_adaptive(
    py::array_t<double> rhs_in,
    py::array_t<double> boundary_values,
    double x_lo, double x_hi,
    double y_lo, double y_hi,
    double z_lo, double z_hi,
    double tol,
    int nghost,
    bool fortran_order_rhs,
    bool fortran_order_bc)
{
    check_3d_array(rhs_in);
    check_3d_array(boundary_values);

    py::buffer_info rhs_buf = rhs_in.request();
    py::buffer_info bc_buf = boundary_values.request();

    // Assuming shape is (nz, ny, nx) for C-order: [k, j, i]
    int nx = static_cast<int>(rhs_buf.shape[2]);
    int ny = static_cast<int>(rhs_buf.shape[1]);
    int nz = static_cast<int>(rhs_buf.shape[0]);

    // Create output numpy array for phi with same shape and layout as input (C order)
    py::array_t<double> phi_out({nz, ny, nx});
    auto phi_buf = phi_out.request();

    // Build AMReX geometry
    IntVect dom_lo(0, 0, 0);
    IntVect dom_hi(nx-1, ny-1, nz-1);
    Box domain(dom_lo, dom_hi);

    RealBox real_box({AMREX_D_DECL(x_lo, y_lo, z_lo)},
                     {AMREX_D_DECL(x_hi, y_hi, z_hi)});
    int coord = 0;
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};

    Geometry geom;
    geom.define(domain, real_box, coord, is_periodic);

    // Single-level BoxArray / DistributionMapping
    BoxArray ba(domain);
    ba.maxSize(32); // chunking
    DistributionMapping dm(ba);

    // --- Coarse level (as before) ---
    IntVect ref_ratio(AMREX_D_DECL(2,2,2));
    int nlevs = 2;

    Vector<Geometry>           geomVec(nlevs);
    Vector<BoxArray>           baVec(nlevs);
    Vector<DistributionMapping> dmVec(nlevs);

    // Level 0: coarse
    geomVec[0] = geom;
    baVec[0]   = ba;     // your existing coarse BoxArray
    dmVec[0]   = dm;

    // Level 1: fine (sphere region only)
    const Real sphere_center[3] = {
        0.5*(x_lo+x_hi), 0.5*(y_lo+y_hi), 0.5*(z_lo+z_hi)
    };
    const Real sphere_radius = 0.25 * (x_hi - x_lo); // 25% of domain

    BoxArray  fine_ba = make_refined_ba(geom, domain, sphere_center,
                                        sphere_radius, /*max_grid_size=*/16);
    Geometry  fine_geom;
    {
        Box fine_domain(IntVect(0),
                        IntVect(AMREX_D_DECL(2*nx-1, 2*ny-1, 2*nz-1)));
        fine_geom.define(fine_domain, real_box, coord, is_periodic);
    }
    DistributionMapping fine_dm(fine_ba);

    geomVec[1] = fine_geom;
    baVec[1]   = fine_ba;
    dmVec[1]   = fine_dm;

    // --- MultiFabs for both levels ---
    MultiFab mf_phi_0(baVec[0], dmVec[0], 1, nghost);
    MultiFab mf_phi_1(baVec[1], dmVec[1], 1, nghost);
    MultiFab mf_rhs_0(baVec[0], dmVec[0], 1, 0);
    MultiFab mf_rhs_1(baVec[1], dmVec[1], 1, 0);

    mf_phi_0.setVal(0.0);
    mf_phi_1.setVal(0.0);

    // Copy Python RHS -> mf_rhs_0
    for (MFIter mfi(mf_rhs_0); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mf_rhs_0.array(mfi);
        const Real* dx      = geom.CellSize();
        const Real* prob_lo = geom.ProbLo();
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
            if (!fortran_order_rhs) {
                size_t idx = static_cast<size_t>(k)*ny*nx + static_cast<size_t>(j)*nx + static_cast<size_t>(i);
                arr(i,j,k) = *(((double*)rhs_buf.ptr) + idx);
            } else {
                size_t idx = static_cast<size_t>(i)*ny*nz + static_cast<size_t>(j)*nz + static_cast<size_t>(k);
                arr(i,j,k) = *(((double*)rhs_buf.ptr) + idx);
            }
        }
    }

    // Set Dirichlet ghost cells in mf_phi_0 from boundary_values
    // AMReX reads the ghost cell *outside* the domain face for Dirichlet BCs
    for (MFIter mfi(mf_phi_0); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mf_phi_0.array(mfi);

        // x lo face: ghost cell at i = -1
        if (bx.smallEnd(0) == 0) {
            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                size_t idx = !fortran_order_bc
                    ? static_cast<size_t>(k)*ny*nx + static_cast<size_t>(j)*nx + 0
                    : static_cast<size_t>(0)*ny*nz + static_cast<size_t>(j)*nz + k;
                arr(-1, j, k) = *(((double*)bc_buf.ptr) + idx);
            }
        }

        // x hi face: ghost cell at i = nx
        if (bx.bigEnd(0) == nx-1) {
            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                size_t idx = !fortran_order_bc
                    ? static_cast<size_t>(k)*ny*nx + static_cast<size_t>(j)*nx + (nx-1)
                    : static_cast<size_t>(nx-1)*ny*nz + static_cast<size_t>(j)*nz + k;
                arr(nx, j, k) = *(((double*)bc_buf.ptr) + idx);
            }
        }

        // y lo face: ghost cell at j = -1
        if (bx.smallEnd(1) == 0) {
            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                size_t idx = !fortran_order_bc
                    ? static_cast<size_t>(k)*ny*nx + 0 + i
                    : static_cast<size_t>(i)*ny*nz + 0 + k;
                arr(i, -1, k) = *(((double*)bc_buf.ptr) + idx);
            }
        }

        // y hi face: ghost cell at j = ny
        if (bx.bigEnd(1) == ny-1) {
            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                size_t idx = !fortran_order_bc
                    ? static_cast<size_t>(k)*ny*nx + static_cast<size_t>(ny-1)*nx + i
                    : static_cast<size_t>(i)*ny*nz + static_cast<size_t>(ny-1)*nz + k;
                arr(i, ny, k) = *(((double*)bc_buf.ptr) + idx);
            }
        }

        // z lo face: ghost cell at k = -1
        if (bx.smallEnd(2) == 0) {
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                size_t idx = !fortran_order_bc
                    ? 0 + static_cast<size_t>(j)*nx + i
                    : static_cast<size_t>(i)*ny*nz + static_cast<size_t>(j)*nz + 0;
                arr(i, j, -1) = *(((double*)bc_buf.ptr) + idx);
            }
        }

        // z hi face: ghost cell at k = nz
        if (bx.bigEnd(2) == nz-1) {
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                size_t idx = !fortran_order_bc
                    ? static_cast<size_t>(nz-1)*ny*nx + static_cast<size_t>(j)*nx + i
                    : static_cast<size_t>(i)*ny*nz + static_cast<size_t>(j)*nz + (nz-1);
                arr(i, j, nz) = *(((double*)bc_buf.ptr) + idx);
            }
        }
    }

    // Interpolater (e.g., cell-centered linear)
    amrex::CellConservativeLinear interp;

    // Fill coarse RHS ghost cells before interpolating
    mf_rhs_0.FillBoundary(geomVec[0].periodicity());

    Vector<BCRec> bcrec(1);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        bcrec[0].setLo(idim, BCType::ext_dir);
        bcrec[0].setHi(idim, BCType::ext_dir);
    }

    using BCFunc = std::function<void(amrex::MultiFab&, int, int,
        amrex::IntVect const&,
        amrex::Real, int)>;

    BCFunc noopBC = [](amrex::MultiFab&, int, int,
        amrex::IntVect const&,
        amrex::Real, int) {};
            
    // Fill fine RHS via interpolation from coarse
    amrex::InterpFromCoarseLevel(
        mf_rhs_1, 0.0,
        mf_rhs_0, 0, 0, 1,
        geomVec[0], geomVec[1],
        noopBC, 0,
        noopBC, 0,
        ref_ratio, &interp,
        bcrec, 0
    );

    // --- MLPoisson with 2 levels ---
    LPInfo info;
    MLPoisson mlpoisson(geomVec, baVec, dmVec, info);

    mlpoisson.setDomainBC(
        {AMREX_D_DECL(LinOpBCType::Dirichlet, LinOpBCType::Dirichlet, LinOpBCType::Dirichlet)},
        {AMREX_D_DECL(LinOpBCType::Dirichlet, LinOpBCType::Dirichlet, LinOpBCType::Dirichlet)});

    mlpoisson.setLevelBC(0, &mf_phi_0);
    mlpoisson.setLevelBC(1, nullptr);

    MLMG mlmg(mlpoisson);
    mlmg.setVerbose(0);
    mlmg.solve({&mf_phi_0, &mf_phi_1}, {&mf_rhs_0, &mf_rhs_1}, tol, tol);

    amrex::average_down(mf_phi_1, mf_phi_0, geomVec[1], geomVec[0],
        0, 1, ref_ratio);

    // Copy mf_phi_0 -> phi_out (C-order)
    for (MFIter mfi(mf_phi_0); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mf_phi_0.const_array(mfi);
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                    size_t idx = static_cast<size_t>(k)*ny*nx +
                                static_cast<size_t>(j)*nx +
                                static_cast<size_t>(i);
                    *(((double*)phi_buf.ptr) + idx) = arr(i,j,k);
                }
            }
        }
    }

    return phi_out;
}

// py::array_t<double> apply_grad(
//     py::array_t<double> x_in,
//     py::array_t<double> boundary_values,
//     double x_lo, double x_hi,
//     double y_lo, double y_hi,
//     double z_lo, double z_hi,
//     int nghost,
//     bool fortran_order_x,
//     bool fortran_order_bc)
// {
//     check_3d_array(x_in);

//     py::buffer_info x_buf = x_in.request();
//     py::buffer_info bc_buf = boundary_values.request();

//     // Assuming shape is (nz, ny, nx) for C-order: [k, j, i]
//     int nx = static_cast<int>(x_buf.shape[2]);
//     int ny = static_cast<int>(x_buf.shape[1]);
//     int nz = static_cast<int>(x_buf.shape[0]);

//     // Create output numpy array for Grad x with same shape and layout as input (C order)
//     py::array_t<double> grad_out({nz, ny, nx});
//     auto grad_buf = grad_out.request();

//     // Build AMReX geometry
//     IntVect dom_lo(0, 0, 0);
//     IntVect dom_hi(nx-1, ny-1, nz-1);
//     Box domain(dom_lo, dom_hi);

//     RealBox real_box({AMREX_D_DECL(x_lo, y_lo, z_lo)},
//                      {AMREX_D_DECL(x_hi, y_hi, z_hi)});
//     int coord = 0;
//     Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};

//     Geometry geom;
//     geom.define(domain, real_box, coord, is_periodic);

//     // Single-level BoxArray / DistributionMapping
//     BoxArray ba(domain);
//     ba.maxSize(32); // chunking
//     DistributionMapping dm(ba);

//     // --- Coarse level (as before) ---
//     IntVect ref_ratio(AMREX_D_DECL(2,2,2));
//     int nlevs = 1; // CHANGE FOR MORE REFINEMENT LEVELS

//     Vector<Geometry>           geomVec(nlevs);
//     Vector<BoxArray>           baVec(nlevs);
//     Vector<DistributionMapping> dmVec(nlevs);

//     // Level 0: coarse (ONLY COARSE FOR NOW, CAN CHANGE THIS LATER)
//     geomVec[0] = geom;
//     baVec[0]   = ba;     // your existing coarse BoxArray
//     dmVec[0]   = dm;

//     // --- MultiFabs ---
//     MultiFab mf_grad_0(baVec[0], dmVec[0], 1, nghost);
//     MultiFab mf_x_0(baVec[0], dmVec[0], 1, nghost);

//     Vector<Array<MultiFab, AMREX_SPACEDIM>> grad(nlevs);
//     for (int lev = 0; lev < nlevs; ++lev) {
//         for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
//             // Create MultiFabs on faces with 0 ghost cells
//             grad[lev][idim].define(amrex::convert(grids[lev], amrex::IntVect::TheDimensionVector(idim)),
//                                 dmVec[lev], 1, 0);
//         }
//     }

//     mf_grad_0.setVal(0.0);
//     mf_lap_0.setVal(0.0);

//     // Copy Python x_in -> mf_x_0
//     for (MFIter mfi(mf_x_0); mfi.isValid(); ++mfi) {
//         const Box& bx = mfi.validbox();
//         auto arr = mf_x_0.array(mfi);
//         const Real* dx      = geom.CellSize();
//         const Real* prob_lo = geom.ProbLo();
//         for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
//         for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
//         for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
//             if (!fortran_order_x) {
//                 size_t idx = static_cast<size_t>(k)*ny*nx + static_cast<size_t>(j)*nx + static_cast<size_t>(i);
//                 arr(i,j,k) = *(((double*)x_buf.ptr) + idx);
//             } else {
//                 size_t idx = static_cast<size_t>(i)*ny*nz + static_cast<size_t>(j)*nz + static_cast<size_t>(k);
//                 arr(i,j,k) = *(((double*)x_buf.ptr) + idx);
//             }
//         }
//     }

//     // Set Dirichlet ghost cells in mf_x_0 from boundary_values
//     // AMReX reads the ghost cell *outside* the domain face for Dirichlet BCs
//     for (MFIter mfi(mf_x_0); mfi.isValid(); ++mfi) {
//         const Box& bx = mfi.validbox();
//         auto arr = mf_x_0.array(mfi);

//         // x lo face: ghost cell at i = -1
//         if (bx.smallEnd(0) == 0) {
//             for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
//             for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
//                 size_t idx = !fortran_order_bc
//                     ? static_cast<size_t>(k)*ny*nx + static_cast<size_t>(j)*nx + 0
//                     : static_cast<size_t>(0)*ny*nz + static_cast<size_t>(j)*nz + k;
//                 arr(-1, j, k) = *(((double*)bc_buf.ptr) + idx);
//             }
//         }

//         // x hi face: ghost cell at i = nx
//         if (bx.bigEnd(0) == nx-1) {
//             for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
//             for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
//                 size_t idx = !fortran_order_bc
//                     ? static_cast<size_t>(k)*ny*nx + static_cast<size_t>(j)*nx + (nx-1)
//                     : static_cast<size_t>(nx-1)*ny*nz + static_cast<size_t>(j)*nz + k;
//                 arr(nx, j, k) = *(((double*)bc_buf.ptr) + idx);
//             }
//         }

//         // y lo face: ghost cell at j = -1
//         if (bx.smallEnd(1) == 0) {
//             for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
//             for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
//                 size_t idx = !fortran_order_bc
//                     ? static_cast<size_t>(k)*ny*nx + 0 + i
//                     : static_cast<size_t>(i)*ny*nz + 0 + k;
//                 arr(i, -1, k) = *(((double*)bc_buf.ptr) + idx);
//             }
//         }

//         // y hi face: ghost cell at j = ny
//         if (bx.bigEnd(1) == ny-1) {
//             for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
//             for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
//                 size_t idx = !fortran_order_bc
//                     ? static_cast<size_t>(k)*ny*nx + static_cast<size_t>(ny-1)*nx + i
//                     : static_cast<size_t>(i)*ny*nz + static_cast<size_t>(ny-1)*nz + k;
//                 arr(i, ny, k) = *(((double*)bc_buf.ptr) + idx);
//             }
//         }

//         // z lo face: ghost cell at k = -1
//         if (bx.smallEnd(2) == 0) {
//             for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
//             for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
//                 size_t idx = !fortran_order_bc
//                     ? 0 + static_cast<size_t>(j)*nx + i
//                     : static_cast<size_t>(i)*ny*nz + static_cast<size_t>(j)*nz + 0;
//                 arr(i, j, -1) = *(((double*)bc_buf.ptr) + idx);
//             }
//         }

//         // z hi face: ghost cell at k = nz
//         if (bx.bigEnd(2) == nz-1) {
//             for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
//             for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
//                 size_t idx = !fortran_order_bc
//                     ? static_cast<size_t>(nz-1)*ny*nx + static_cast<size_t>(j)*nx + i
//                     : static_cast<size_t>(i)*ny*nz + static_cast<size_t>(j)*nz + (nz-1);
//                 arr(i, j, nz) = *(((double*)bc_buf.ptr) + idx);
//             }
//         }
//     }

//     // --- MLPoisson  ---
//     LPInfo info;
//     MLPoisson mlpoisson(geomVec, baVec, dmVec, info);

//     mlpoisson.setDomainBC(
//         {AMREX_D_DECL(LinOpBCType::Dirichlet, LinOpBCType::Dirichlet, LinOpBCType::Dirichlet)},
//         {AMREX_D_DECL(LinOpBCType::Dirichlet, LinOpBCType::Dirichlet, LinOpBCType::Dirichlet)});

//     // ???
//     mlpoisson.setLevelBC(0, &mf_x_0);

//     MLMG mlmg(mlpoisson);
//     mlmg.setVerbose(0);
//     mlmg.apply({&mf_lap_0}, {&mf_x_0});

//     // Copy mf_phi_0 -> phi_out (C-order)
//     for (MFIter mfi(mf_lap_0); mfi.isValid(); ++mfi) {
//         const Box& bx = mfi.validbox();
//         auto arr = mf_lap_0.const_array(mfi);
//         for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
//             for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
//                 for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
//                     size_t idx = static_cast<size_t>(k)*ny*nx +
//                                 static_cast<size_t>(j)*nx +
//                                 static_cast<size_t>(i);
//                     *(((double*)lap_buf.ptr) + idx) = arr(i,j,k);
//                 }
//             }
//         }
//     }

//     return lap_out;
// }

py::array_t<double> solve_poisson(py::array_t<double> rhs_in,
                                  double x_lo = 0.0, double x_hi = 2.0*M_PI,
                                  double y_lo = 0.0, double y_hi = 2.0*M_PI,
                                  double z_lo = 0.0, double z_hi = 2.0*M_PI,
                                  double tol = 1e-10,
                                  int nghost = 1,
                                  bool fortran_order_rhs = false)
{
    check_3d_array(rhs_in);
    py::buffer_info buf = rhs_in.request();

    // Assuming shape is (nz, ny, nx) for C-order: [k, j, i]
    int nx = static_cast<int>(buf.shape[2]);
    int ny = static_cast<int>(buf.shape[1]);
    int nz = static_cast<int>(buf.shape[0]);

    // Create output numpy array for phi with same shape and layout as input (C order)
    py::array_t<double> phi_out({nz, ny, nx});
    auto phi_buf = phi_out.request();

    // Build AMReX geometry
    IntVect dom_lo(0, 0, 0);
    IntVect dom_hi(nx-1, ny-1, nz-1);
    Box domain(dom_lo, dom_hi);

    RealBox real_box({AMREX_D_DECL(x_lo, y_lo, z_lo)},
                     {AMREX_D_DECL(x_hi, y_hi, z_hi)});
    int coord = 0;
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};

    Geometry geom;
    geom.define(domain, real_box, coord, is_periodic);

    // Single-level BoxArray / DistributionMapping
    BoxArray ba(domain);
    ba.maxSize(32); // chunking
    DistributionMapping dm(ba);

    // MultiFabs
    int ncomp = 1;
    MultiFab mf_rhs(ba, dm, ncomp, 0);
    MultiFab mf_phi(ba, dm, ncomp, nghost);

    mf_phi.setVal(0.0);

    // Copy Python RHS -> mf_rhs
    // Support both C-order (row-major) and Fortran-order ravel: user sets fortran_order_rhs flag
    // We assume python array layout is (nz, ny, nx) with indexes [k,j,i] => [dim0, dim1, dim2]
    for (MFIter mfi(mf_rhs); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mf_rhs.array(mfi);
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                    // map (i,j,k) -> python index (dim0=k, dim1=j, dim2=i)
                    // C-contiguous numpy: data[k*ny*nx + j*nx + i]
                    // Fortran-order flattening: data[i*ny*nz + j*nz + k]
                    if (!fortran_order_rhs) {
                        size_t idx = static_cast<size_t>(k)*ny*nx + 
                                    static_cast<size_t>(j)*nx + 
                                    static_cast<size_t>(i);
                        arr(i,j,k) = *(((double*)buf.ptr) + idx);
                    } else {
                        size_t idx = static_cast<size_t>(i)*ny*nz + 
                                    static_cast<size_t>(j)*nz + 
                                    static_cast<size_t>(k);
                        arr(i,j,k) = *(((double*)buf.ptr) + idx);
                    }
                }
            }
        }
    }

    // Build solver operator and MLMG (single-level Poisson)
    LPInfo info;
    info.setMaxCoarseningLevel(0);

    Vector<Geometry> geomVec(1);
    geomVec[0] = geom;
    Vector<BoxArray> baVec(1);
    baVec[0] = ba;
    Vector<DistributionMapping> dmVec(1);
    dmVec[0] = dm;

    MLPoisson mlpoisson(geomVec, baVec, dmVec, info);
    mlpoisson.setDomainBC({AMREX_D_DECL(LinOpBCType::Dirichlet, 
                                        LinOpBCType::Dirichlet, 
                                        LinOpBCType::Dirichlet)},
                          {AMREX_D_DECL(LinOpBCType::Dirichlet, 
                                        LinOpBCType::Dirichlet, 
                                        LinOpBCType::Dirichlet)});

    mlpoisson.setLevelBC(0, nullptr);

    MLMG mlmg(mlpoisson);
    mlmg.setVerbose(0);

    // Solve: mlmg solves A * phi = rhs for Laplacian operator as configured by MLPoisson.
    // Note: AMReX uses -div(a grad) form. If you need signs matched to your old code,
    // you may need to flip sign of rhs or postprocess.
    mlmg.solve({&mf_phi}, {&mf_rhs}, tol, tol);

    // Copy mf_phi -> phi_out (C-order)
    for (MFIter mfi(mf_phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mf_phi.const_array(mfi);
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                    size_t idx = static_cast<size_t>(k)*ny*nx + 
                                static_cast<size_t>(j)*nx + 
                                static_cast<size_t>(i);
                    *(((double*)phi_buf.ptr) + idx) = arr(i,j,k);
                }
            }
        }
    }

    return phi_out;
}

std::pair<py::array_t<double>, py::array_t<double>> apply_poisson_double_grid(
    py::array_t<double> x_in_coarse,
    py::array_t<double> x_in_fine,
    py::array_t<double> boundary_values,
    double x_lo, double x_hi,
    double y_lo, double y_hi,
    double z_lo, double z_hi,
    int nghost,
    bool fortran_order_x,
    bool fortran_order_bc)
{
    check_3d_array(x_in_coarse);
    check_3d_array(x_in_fine);
    check_3d_array(boundary_values);

    py::buffer_info x_coarse_buf = x_in_coarse.request();
    py::buffer_info x_fine_buf   = x_in_fine.request();
    py::buffer_info bc_buf         = boundary_values.request();

    int nx_coarse = static_cast<int>(x_coarse_buf.shape[2]);
    int ny_coarse = static_cast<int>(x_coarse_buf.shape[1]);
    int nz_coarse = static_cast<int>(x_coarse_buf.shape[0]);

    // FIX: fine patch dimensions come from rhs_in_fine, which covers only
    // the refined sub-region (not the full refined domain).
    int nx_fine_patch = static_cast<int>(x_fine_buf.shape[2]);
    int ny_fine_patch = static_cast<int>(x_fine_buf.shape[1]);
    int nz_fine_patch = static_cast<int>(x_fine_buf.shape[0]);

    // Output arrays
    py::array_t<double> lap_out_coarse({nz_coarse, ny_coarse, nx_coarse});

    // FIX: fine output has same shape as fine patch input, not full refined domain
    py::array_t<double> lap_out_fine({nz_fine_patch, ny_fine_patch, nx_fine_patch});

    auto lap_coarse_buf = lap_out_coarse.request();
    auto lap_fine_buf   = lap_out_fine.request();

    // Build coarse geometry
    IntVect dom_lo(0, 0, 0);
    IntVect dom_hi(nx_coarse-1, ny_coarse-1, nz_coarse-1);
    Box domain(dom_lo, dom_hi);

    RealBox real_box({AMREX_D_DECL(x_lo, y_lo, z_lo)},
                     {AMREX_D_DECL(x_hi, y_hi, z_hi)});
    int coord = 0;
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};

    Geometry geom;
    geom.define(domain, real_box, coord, is_periodic);

    BoxArray ba(domain);
    ba.maxSize(32);
    DistributionMapping dm(ba);

    IntVect ref_ratio(AMREX_D_DECL(2,2,2));
    int nlevs = 2;

    Vector<Geometry>            geomVec(nlevs);
    Vector<BoxArray>            baVec(nlevs);
    Vector<DistributionMapping> dmVec(nlevs);

    geomVec[0] = geom;
    baVec[0]   = ba;
    dmVec[0]   = dm;

    // Fine level geometry and BoxArray
    const Real cube_center[3] = {
        0.5*(x_lo+x_hi), 0.5*(y_lo+y_hi), 0.5*(z_lo+z_hi)
    };
    const Real cube_half_widths[3] = {
        0.25*(x_hi-x_lo), 0.25*(y_hi-y_lo), 0.25*(z_hi-z_lo)
    };

    BoxArray fine_ba = make_refined_ba_cube(geom, domain, cube_center, cube_half_widths, 16);

    // FIX: fine_domain spans the full refined index space (coarse * ref_ratio),
    // but fine_ba only covers the central patch. We need the full fine domain
    // for Geometry so AMReX can compute cell sizes correctly.
    int nx_fine_dom = nx_coarse * ref_ratio[0];
    int ny_fine_dom = ny_coarse * ref_ratio[1];
    int nz_fine_dom = nz_coarse * ref_ratio[2];

    Geometry fine_geom;
    {
        Box fine_domain(IntVect(0),
        IntVect(AMREX_D_DECL(nx_fine_dom-1, ny_fine_dom-1, nz_fine_dom-1)));
        fine_geom.define(fine_domain, real_box, coord, is_periodic);
    }
    DistributionMapping fine_dm(fine_ba);

    geomVec[1] = fine_geom;
    baVec[1]   = fine_ba;
    dmVec[1]   = fine_dm;

    // MultiFabs
    MultiFab mf_x_0(baVec[0], dmVec[0], 1, nghost);
    MultiFab mf_x_1(baVec[1], dmVec[1], 1, nghost);
    MultiFab mf_lap_0(baVec[0], dmVec[0], 1, 0);
    MultiFab mf_lap_1(baVec[1], dmVec[1], 1, 0);

    mf_lap_0.setVal(0.0);
    mf_lap_1.setVal(0.0);

    // Copy coarse x_in
    for (MFIter mfi(mf_x_0); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mf_x_0.array(mfi);
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
            if (!fortran_order_x) {
                size_t idx = (size_t)k*ny_coarse*nx_coarse + (size_t)j*nx_coarse + i;
                arr(i,j,k) = *(((double*)x_coarse_buf.ptr) + idx);
            } else {
                size_t idx = (size_t)i*ny_coarse*nz_coarse + (size_t)j*nz_coarse + k;
                arr(i,j,k) = *(((double*)x_coarse_buf.ptr) + idx);
            }
        }
    }

    // Before the loop, store the patch smallEnd (same for all boxes in the patch)
    Box patch_box = fine_ba.minimalBox();
    IntVect patch_lo = patch_box.smallEnd();

    // Copy fine x_in using zero-based offsets into the patch buffer
    for (MFIter mfi(mf_x_1); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mf_x_1.array(mfi);
        const IntVect lo = bx.smallEnd(); // offset into the patch array
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
            int li = i - patch_lo[0];
            int lj = j - patch_lo[1];
            int lk = k - patch_lo[2];
            if (!fortran_order_x) {
                size_t idx = (size_t)lk*ny_fine_patch*nx_fine_patch + (size_t)lj*nx_fine_patch + li;
                arr(i,j,k) = *(((double*)x_fine_buf.ptr) + idx);
            } else {
                size_t idx = (size_t)li*ny_fine_patch*nz_fine_patch + (size_t)lj*nz_fine_patch + lk;
                arr(i,j,k) = *(((double*)x_fine_buf.ptr) + idx);
            }
        }
    }

    // Set Dirichlet ghost cells in mf_phi_0 (unchanged from original)
    for (MFIter mfi(mf_x_0); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mf_x_0.array(mfi);

        if (bx.smallEnd(0) == 0) {
            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                size_t idx = !fortran_order_bc
                    ? (size_t)k*ny_coarse*nx_coarse + (size_t)j*nx_coarse + 0
                    : (size_t)0*ny_coarse*nz_coarse + (size_t)j*nz_coarse + k;
                arr(-1, j, k) = *(((double*)bc_buf.ptr) + idx);
            }
        }
        if (bx.bigEnd(0) == nx_coarse-1) {
            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                size_t idx = !fortran_order_bc
                    ? (size_t)k*ny_coarse*nx_coarse + (size_t)j*nx_coarse + (nx_coarse-1)
                    : (size_t)(nx_coarse-1)*ny_coarse*nz_coarse + (size_t)j*nz_coarse + k;
                arr(nx_coarse, j, k) = *(((double*)bc_buf.ptr) + idx);
            }
        }
        if (bx.smallEnd(1) == 0) {
            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                size_t idx = !fortran_order_bc
                    ? (size_t)k*ny_coarse*nx_coarse + 0 + i
                    : (size_t)i*ny_coarse*nz_coarse + 0 + k;
                arr(i, -1, k) = *(((double*)bc_buf.ptr) + idx);
            }
        }
        if (bx.bigEnd(1) == ny_coarse-1) {
            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                size_t idx = !fortran_order_bc
                    ? (size_t)k*ny_coarse*nx_coarse + (size_t)(ny_coarse-1)*nx_coarse + i
                    : (size_t)i*ny_coarse*nz_coarse + (size_t)(ny_coarse-1)*nz_coarse + k;
                arr(i, ny_coarse, k) = *(((double*)bc_buf.ptr) + idx);
            }
        }
        if (bx.smallEnd(2) == 0) {
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                size_t idx = !fortran_order_bc
                    ? 0 + (size_t)j*nx_coarse + i
                    : (size_t)i*ny_coarse*nz_coarse + (size_t)j*nz_coarse + 0;
                arr(i, j, -1) = *(((double*)bc_buf.ptr) + idx);
            }
        }
        if (bx.bigEnd(2) == nz_coarse-1) {
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                size_t idx = !fortran_order_bc
                    ? (size_t)(nz_coarse-1)*ny_coarse*nx_coarse + (size_t)j*nx_coarse + i
                    : (size_t)i*ny_coarse*nz_coarse + (size_t)j*nz_coarse + (nz_coarse-1);
                arr(i, j, nz_coarse) = *(((double*)bc_buf.ptr) + idx);
            }
        }
    }

    // --- MLPoisson  ---
    LPInfo info;
    MLPoisson mlpoisson(geomVec, baVec, dmVec, info);

    mlpoisson.setDomainBC(
        {AMREX_D_DECL(LinOpBCType::Dirichlet, LinOpBCType::Dirichlet, LinOpBCType::Dirichlet)},
        {AMREX_D_DECL(LinOpBCType::Dirichlet, LinOpBCType::Dirichlet, LinOpBCType::Dirichlet)});

    mlpoisson.setLevelBC(0, &mf_x_0);
    mlpoisson.setLevelBC(1, nullptr);

    MLMG mlmg(mlpoisson);
    mlmg.setVerbose(0);
    mlmg.apply({&mf_lap_0, &mf_lap_1}, {&mf_x_0, &mf_x_1});

    // Copy coarse solution out
    for (MFIter mfi(mf_lap_0); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mf_lap_0.const_array(mfi);
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
            size_t idx = (size_t)k*ny_coarse*nx_coarse + (size_t)j*nx_coarse + i;
            *(((double*)lap_coarse_buf.ptr) + idx) = arr(i,j,k);
        }
    }

    // FIX: Copy fine solution out using the same zero-based offset logic.
    for (MFIter mfi(mf_lap_1); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mf_lap_1.const_array(mfi);
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
            int li = i - patch_lo[0];
            int lj = j - patch_lo[1];
            int lk = k - patch_lo[2];
            size_t idx = (size_t)lk*ny_fine_patch*nx_fine_patch + (size_t)lj*nx_fine_patch + li;
            *(((double*)lap_fine_buf.ptr) + idx) = arr(i,j,k);
        }
    }

    return {lap_out_coarse, lap_out_fine};
}

std::pair<py::array_t<double>, py::array_t<double>> solve_poisson_adaptive_double_grid(
    py::array_t<double> rhs_in_coarse,
    py::array_t<double> rhs_in_fine,
    py::array_t<double> boundary_values,
    double x_lo, double x_hi,
    double y_lo, double y_hi,
    double z_lo, double z_hi,
    double tol,
    int nghost,
    bool fortran_order_rhs,
    bool fortran_order_bc)
{
    check_3d_array(rhs_in_coarse);
    check_3d_array(rhs_in_fine);
    check_3d_array(boundary_values);

    py::buffer_info rhs_coarse_buf = rhs_in_coarse.request();
    py::buffer_info rhs_fine_buf   = rhs_in_fine.request();
    py::buffer_info bc_buf         = boundary_values.request();

    int nx_coarse = static_cast<int>(rhs_coarse_buf.shape[2]);
    int ny_coarse = static_cast<int>(rhs_coarse_buf.shape[1]);
    int nz_coarse = static_cast<int>(rhs_coarse_buf.shape[0]);

    // FIX: fine patch dimensions come from rhs_in_fine, which covers only
    // the refined sub-region (not the full refined domain).
    int nx_fine_patch = static_cast<int>(rhs_fine_buf.shape[2]);
    int ny_fine_patch = static_cast<int>(rhs_fine_buf.shape[1]);
    int nz_fine_patch = static_cast<int>(rhs_fine_buf.shape[0]);

    // Output arrays
    py::array_t<double> phi_out_coarse({nz_coarse, ny_coarse, nx_coarse});

    // FIX: fine output has same shape as fine patch input, not full refined domain
    py::array_t<double> phi_out_fine({nz_fine_patch, ny_fine_patch, nx_fine_patch});

    auto phi_coarse_buf = phi_out_coarse.request();
    auto phi_fine_buf   = phi_out_fine.request();

    // Build coarse geometry
    IntVect dom_lo(0, 0, 0);
    IntVect dom_hi(nx_coarse-1, ny_coarse-1, nz_coarse-1);
    Box domain(dom_lo, dom_hi);

    RealBox real_box({AMREX_D_DECL(x_lo, y_lo, z_lo)},
                     {AMREX_D_DECL(x_hi, y_hi, z_hi)});
    int coord = 0;
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};

    Geometry geom;
    geom.define(domain, real_box, coord, is_periodic);

    BoxArray ba(domain);
    ba.maxSize(32);
    DistributionMapping dm(ba);

    IntVect ref_ratio(AMREX_D_DECL(2,2,2));
    int nlevs = 2;

    Vector<Geometry>            geomVec(nlevs);
    Vector<BoxArray>            baVec(nlevs);
    Vector<DistributionMapping> dmVec(nlevs);

    geomVec[0] = geom;
    baVec[0]   = ba;
    dmVec[0]   = dm;

    // Fine level geometry and BoxArray
    const Real cube_center[3] = {
        0.5*(x_lo+x_hi), 0.5*(y_lo+y_hi), 0.5*(z_lo+z_hi)
    };
    const Real cube_half_widths[3] = {
        0.25*(x_hi-x_lo), 0.25*(y_hi-y_lo), 0.25*(z_hi-z_lo)
    };

    BoxArray fine_ba = make_refined_ba_cube(geom, domain, cube_center, cube_half_widths, 16);

    // FIX: fine_domain spans the full refined index space (coarse * ref_ratio),
    // but fine_ba only covers the central patch. We need the full fine domain
    // for Geometry so AMReX can compute cell sizes correctly.
    int nx_fine_dom = nx_coarse * ref_ratio[0];
    int ny_fine_dom = ny_coarse * ref_ratio[1];
    int nz_fine_dom = nz_coarse * ref_ratio[2];

    Geometry fine_geom;
    {
        Box fine_domain(IntVect(0),
        IntVect(AMREX_D_DECL(nx_fine_dom-1, ny_fine_dom-1, nz_fine_dom-1)));
        fine_geom.define(fine_domain, real_box, coord, is_periodic);
    }
    DistributionMapping fine_dm(fine_ba);

    geomVec[1] = fine_geom;
    baVec[1]   = fine_ba;
    dmVec[1]   = fine_dm;

    // MultiFabs
    MultiFab mf_phi_0(baVec[0], dmVec[0], 1, nghost);
    MultiFab mf_phi_1(baVec[1], dmVec[1], 1, nghost);
    MultiFab mf_rhs_0(baVec[0], dmVec[0], 1, 0);
    MultiFab mf_rhs_1(baVec[1], dmVec[1], 1, 0);

    mf_phi_0.setVal(0.0);
    mf_phi_1.setVal(0.0);

    // Copy coarse RHS (unchanged from original)
    for (MFIter mfi(mf_rhs_0); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mf_rhs_0.array(mfi);
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
            if (!fortran_order_rhs) {
                size_t idx = (size_t)k*ny_coarse*nx_coarse + (size_t)j*nx_coarse + i;
                arr(i,j,k) = *(((double*)rhs_coarse_buf.ptr) + idx);
            } else {
                size_t idx = (size_t)i*ny_coarse*nz_coarse + (size_t)j*nz_coarse + k;
                arr(i,j,k) = *(((double*)rhs_coarse_buf.ptr) + idx);
            }
        }
    }

    // Before the loop, store the patch smallEnd (same for all boxes in the patch)
    Box patch_box = fine_ba.minimalBox();
    IntVect patch_lo = patch_box.smallEnd();

    // FIX: Copy fine RHS using zero-based offsets into the patch buffer.
    // fine_ba boxes have AMReX indices in the full fine index space, so
    // i,j,k are NOT zero-based. Subtract smallEnd to index into rhs_fine_buf.
    for (MFIter mfi(mf_rhs_1); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mf_rhs_1.array(mfi);
        const IntVect lo = bx.smallEnd(); // offset into the patch array
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
            int li = i - patch_lo[0];
            int lj = j - patch_lo[1];
            int lk = k - patch_lo[2];
            if (!fortran_order_rhs) {
                size_t idx = (size_t)lk*ny_fine_patch*nx_fine_patch + (size_t)lj*nx_fine_patch + li;
                arr(i,j,k) = *(((double*)rhs_fine_buf.ptr) + idx);
            } else {
                size_t idx = (size_t)li*ny_fine_patch*nz_fine_patch + (size_t)lj*nz_fine_patch + lk;
                arr(i,j,k) = *(((double*)rhs_fine_buf.ptr) + idx);
            }
        }
    }

    // Set Dirichlet ghost cells in mf_phi_0 (unchanged from original)
    for (MFIter mfi(mf_phi_0); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mf_phi_0.array(mfi);

        if (bx.smallEnd(0) == 0) {
            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                size_t idx = !fortran_order_bc
                    ? (size_t)k*ny_coarse*nx_coarse + (size_t)j*nx_coarse + 0
                    : (size_t)0*ny_coarse*nz_coarse + (size_t)j*nz_coarse + k;
                arr(-1, j, k) = *(((double*)bc_buf.ptr) + idx);
            }
        }
        if (bx.bigEnd(0) == nx_coarse-1) {
            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                size_t idx = !fortran_order_bc
                    ? (size_t)k*ny_coarse*nx_coarse + (size_t)j*nx_coarse + (nx_coarse-1)
                    : (size_t)(nx_coarse-1)*ny_coarse*nz_coarse + (size_t)j*nz_coarse + k;
                arr(nx_coarse, j, k) = *(((double*)bc_buf.ptr) + idx);
            }
        }
        if (bx.smallEnd(1) == 0) {
            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                size_t idx = !fortran_order_bc
                    ? (size_t)k*ny_coarse*nx_coarse + 0 + i
                    : (size_t)i*ny_coarse*nz_coarse + 0 + k;
                arr(i, -1, k) = *(((double*)bc_buf.ptr) + idx);
            }
        }
        if (bx.bigEnd(1) == ny_coarse-1) {
            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                size_t idx = !fortran_order_bc
                    ? (size_t)k*ny_coarse*nx_coarse + (size_t)(ny_coarse-1)*nx_coarse + i
                    : (size_t)i*ny_coarse*nz_coarse + (size_t)(ny_coarse-1)*nz_coarse + k;
                arr(i, ny_coarse, k) = *(((double*)bc_buf.ptr) + idx);
            }
        }
        if (bx.smallEnd(2) == 0) {
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                size_t idx = !fortran_order_bc
                    ? 0 + (size_t)j*nx_coarse + i
                    : (size_t)i*ny_coarse*nz_coarse + (size_t)j*nz_coarse + 0;
                arr(i, j, -1) = *(((double*)bc_buf.ptr) + idx);
            }
        }
        if (bx.bigEnd(2) == nz_coarse-1) {
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                size_t idx = !fortran_order_bc
                    ? (size_t)(nz_coarse-1)*ny_coarse*nx_coarse + (size_t)j*nx_coarse + i
                    : (size_t)i*ny_coarse*nz_coarse + (size_t)j*nz_coarse + (nz_coarse-1);
                arr(i, j, nz_coarse) = *(((double*)bc_buf.ptr) + idx);
            }
        }
    }

    // Solve
    LPInfo info;
    MLPoisson mlpoisson(geomVec, baVec, dmVec, info);

    mlpoisson.setDomainBC(
        {AMREX_D_DECL(LinOpBCType::Dirichlet, LinOpBCType::Dirichlet, LinOpBCType::Dirichlet)},
        {AMREX_D_DECL(LinOpBCType::Dirichlet, LinOpBCType::Dirichlet, LinOpBCType::Dirichlet)});

    mlpoisson.setLevelBC(0, &mf_phi_0);
    mlpoisson.setLevelBC(1, nullptr);

    MLMG mlmg(mlpoisson);
    mlmg.setVerbose(0);
    mlmg.solve({&mf_phi_0, &mf_phi_1}, {&mf_rhs_0, &mf_rhs_1}, tol, tol);

    // Copy coarse solution out (unchanged from original)
    for (MFIter mfi(mf_phi_0); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mf_phi_0.const_array(mfi);
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
            size_t idx = (size_t)k*ny_coarse*nx_coarse + (size_t)j*nx_coarse + i;
            *(((double*)phi_coarse_buf.ptr) + idx) = arr(i,j,k);
        }
    }

    // FIX: Copy fine solution out using the same zero-based offset logic.
    for (MFIter mfi(mf_phi_1); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mf_phi_1.const_array(mfi);
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k)
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
            int li = i - patch_lo[0];
            int lj = j - patch_lo[1];
            int lk = k - patch_lo[2];
            size_t idx = (size_t)lk*ny_fine_patch*nx_fine_patch + (size_t)lj*nx_fine_patch + li;
            *(((double*)phi_fine_buf.ptr) + idx) = arr(i,j,k);
        }
    }

    return {phi_out_coarse, phi_out_fine};
}


PYBIND11_MODULE(amrex_poisson_3d, m) {
    m.doc() = "Minimal AMReX Poisson wrapper for 3D (skeleton)";

    m.def("amrex_init", &amrex_init, 
        py::arg("argv") = py::list(),
        "Initialize AMReX (optionally pass argv list)");
    m.def("amrex_finalize", &amrex_finalize, "Finalize AMReX");
    m.def("solve_poisson", &solve_poisson,
        py::arg("rhs_in"),
        py::arg("x_lo") = 0.0,
        py::arg("x_hi") = 2.0*M_PI,
        py::arg("y_lo") = 0.0,
        py::arg("y_hi") = 2.0*M_PI,
        py::arg("z_lo") = 0.0,
        py::arg("z_hi") = 2.0*M_PI,
        py::arg("tol") = 1e-10,
        py::arg("nghost") = 1,
        py::arg("fortran_order_rhs") = false);
    m.def("solve_poisson_adaptive_double_grid", &solve_poisson_adaptive_double_grid,
        py::arg("rhs_in_coarse"),
        py::arg("rhs_in_fine"),
        py::arg("boundary_values"),
        py::arg("x_lo") = 0.0,
        py::arg("x_hi") = 2.0*M_PI,
        py::arg("y_lo") = 0.0,
        py::arg("y_hi") = 2.0*M_PI,
        py::arg("z_lo") = 0.0,
        py::arg("z_hi") = 2.0*M_PI,
        py::arg("tol") = 1e-10,
        py::arg("nghost") = 1,
        py::arg("fortran_order_rhs") = false,
        py::arg("fortran_order_bc") = false,
        "Solve Poisson equation on both fine and coarse grid");
    m.def("solve_poisson_adaptive", &solve_poisson_adaptive,
        py::arg("rhs_in"),
        py::arg("boundary_values"),
        py::arg("x_lo") = 0.0,
        py::arg("x_hi") = 2.0*M_PI,
        py::arg("y_lo") = 0.0,
        py::arg("y_hi") = 2.0*M_PI,
        py::arg("z_lo") = 0.0,
        py::arg("z_hi") = 2.0*M_PI,
        py::arg("tol") = 1e-10,
        py::arg("nghost") = 1,
        py::arg("fortran_order_rhs") = false,
        py::arg("fortran_order_bc") = false,
        "Solve Poisson equation with AMR refinement around a sphere at the domain center");
    m.def("apply_poisson_double_grid", &apply_poisson_double_grid, 
        py::arg("x_in_coarse"),
        py::arg("x_in_fine"),
        py::arg("boundary_values"),
        py::arg("x_lo") = 0.0,
        py::arg("x_hi") = 2.0*M_PI,
        py::arg("y_lo") = 0.0,
        py::arg("y_hi") = 2.0*M_PI,
        py::arg("z_lo") = 0.0,
        py::arg("z_hi") = 2.0*M_PI,
        py::arg("nghost") = 1,
        py::arg("fortran_order_x") = false,
        py::arg("fortran_order_bc") = false);
}