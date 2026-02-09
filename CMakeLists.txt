cmake_minimum_required(VERSION 3.18)
project(amrex_poisson_3d)

set(CMAKE_CXX_STANDARD 17)
set(AMREX_SPACEDIM 3 CACHE STRING "")

# Find or build AMReX
add_subdirectory(amrex)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Create the Python module
pybind11_add_module(amrex_poisson_3d 3d_poisson.cpp)

target_link_libraries(amrex_poisson_3d PRIVATE amrex)
target_compile_definitions(amrex_poisson_3d PRIVATE AMREX_SPACEDIM=3)