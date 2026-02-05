#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MLMG.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Array.H>
#include <AMReX_ParallelDescriptor.H>

namespace py = pybind11;
using namespace amrex;

// Helper: require 2D arrays for this skeleton
static void check_2d_array(const py::array_t<double>& arr) {
    if (arr.ndim() != 2) {
        throw std::runtime_error("Only 2D arrays supported in this skeleton.");
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

py::array_t<double> solve_poisson(py::array_t<double> rhs_in,
                                  double x_lo = 0.0, double x_hi = 2.0*M_PI,
                                  double y_lo = 0.0, double y_hi = 2.0*M_PI,
                                  double tol = 1e-10,
                                  int nghost = 1,
                                  bool fortran_order_rhs = false)
{
    check_2d_array(rhs_in);
    py::buffer_info buf = rhs_in.request();

    int nx = static_cast<int>(buf.shape[1]); // shape = (ny, nx) if Fortran? We'll treat shape[0]=ny, shape[1]=nx
    int ny = static_cast<int>(buf.shape[0]);

    // Create output numpy array for phi with same shape and layout as input (C order)
    py::array_t<double> phi_out({ny, nx});
    auto phi_buf = phi_out.request();

    // Build AMReX geometry
    IntVect dom_lo(0,0);
    IntVect dom_hi(nx-1, ny-1);
    Box domain(dom_lo, dom_hi);

    RealBox real_box({AMREX_D_DECL(x_lo, y_lo, 0.0)},
                     {AMREX_D_DECL(x_hi, y_hi, 0.0)});
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
    // We assume python array layout is (ny, nx) with indexes [j,i] => [row, col]
    for (MFIter mfi(mf_rhs); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto arr = mf_rhs.array(mfi);
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                // map (i,j) -> python index (row=j, col=i)
                // C-contiguous numpy: data[row*nx + col]
                // Fortran-order flattening: data[col*ny + row]
                if (!fortran_order_rhs) {
                    size_t idx = static_cast<size_t>(j)*nx + static_cast<size_t>(i);
                    arr(i,j,0) = *(((double*)buf.ptr) + idx);
                } else {
                    size_t idx = static_cast<size_t>(i)*ny + static_cast<size_t>(j);
                    arr(i,j,0) = *(((double*)buf.ptr) + idx);
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
    mlpoisson.setDomainBC({AMREX_D_DECL(LinOpBCType::Dirichlet, LinOpBCType::Dirichlet, LinOpBCType::Dirichlet)},
                          {AMREX_D_DECL(LinOpBCType::Dirichlet, LinOpBCType::Dirichlet, LinOpBCType::Dirichlet)});

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
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                size_t idx = static_cast<size_t>(j)*nx + static_cast<size_t>(i);
                *(((double*)phi_buf.ptr) + idx) = arr(i,j,0);
            }
        }
    }

    return phi_out;
}

PYBIND11_MODULE(amrex_poisson, m) {
    m.doc() = "Minimal AMReX Poisson wrapper (skeleton)";

    m.def("amrex_init", &amrex_init, "Initialize AMReX (optionally pass argv list)");
    m.def("amrex_finalize", &amrex_finalize, "Finalize AMReX");
    m.def("solve_poisson", &solve_poisson,
          py::arg("rhs_in"),
          py::arg("x_lo") = 0.0,
          py::arg("x_hi") = 2.0*M_PI,
          py::arg("y_lo") = 0.0,
          py::arg("y_hi") = 2.0*M_PI,
          py::arg("tol") = 1e-10,
          py::arg("nghost") = 1,
          py::arg("fortran_order_rhs") = false);
}