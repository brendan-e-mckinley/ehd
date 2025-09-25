cmake_minimum_required(VERSION 3.20)
project(amrex_poisson LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tell AMReX what we want
set(AMReX_SPACEDIM 2 CACHE STRING "Use 2D build")
set(AMReX_MPI OFF CACHE BOOL "Disable MPI")
set(AMReX_EB ON CACHE BOOL "Enable Embedded Boundaries")

# Add AMReX as a subdirectory (assumes you have AMReX in ./amrex)
add_subdirectory(amrex)

# Find pybind11 (either system-installed or via FetchContent)
find_package(pybind11 CONFIG REQUIRED)

# Build the Python extension module
pybind11_add_module(amrex_poisson amrex_laplacian.cpp)

# Link against AMReX
target_link_libraries(amrex_poisson PRIVATE amrex)

# Define compile-time macros to match AMReX config
target_compile_definitions(amrex_poisson PRIVATE
    AMREX_SPACEDIM=2
    AMREX_USE_EB=0
)

# Optional: silence some warnings from AMReX headers
if (MSVC)
    target_compile_options(amrex_poisson PRIVATE /wd4244 /wd4267)
else()
    target_compile_options(amrex_poisson PRIVATE -Wall -Wextra -Wno-sign-compare)
endif()
