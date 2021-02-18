#-----------------------------------------------------------------------------
# gmslib - Gaussian Mixture Surface Library
# Copyright (c) Adam Celarek 2021 <celarek at cg dot tuwien dot ac dot at>
#               Simon Fraiss 2021
#
# Usage is subject to the terms of the WFP (modified BSD-3-Clause) license.
# See the accompanied LICENSE file or
# https://github.com/rpreiner/gmslib/blob/main/LICENSE
#-----------------------------------------------------------------------------


cmake_minimum_required(VERSION 3.18)

project(gmslib LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ASan:
#string(APPEND CMAKE_CXX_FLAGS " -fno-omit-frame-pointer -fsanitize=address")
#string(APPEND CMAKE_LINKER_FLAGS" -fno-omit-frame-pointer -fsanitize=address")

#target_compile_options(kre -Wall -Wextra -Wpedantic -Werror)
string(APPEND CMAKE_CXX_FLAGS " -ffast-math -fno-finite-math-only")
string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O4 -march=native")

# this will enable the lld (llvm) linker. it's parallel and much faster, but not installed by default.
# if it's not installed, you'll get errors, that openmp or other stuff is not installed (hard to track down)
#string(APPEND CMAKE_EXE_LINKER_FLAGS " -fuse-ld=lld")

set (MY_ANACONDA_PATH "/home/madam/bin/anaconda3")

list(APPEND CMAKE_PREFIX_PATH "${MY_ANACONDA_PATH}/lib/python3.7/site-packages/torch/")

find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)
add_compile_definitions(GMSLIB_CMAKE_TEST_BUILD)
add_compile_definitions(_HAS_STD_BYTE=0)
add_compile_definitions(NOMINMAX=1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


set(MY_INCLUDE_PATHS
    ${MY_ANACONDA_PATH}/include/python3.7m/
    ./ext
    .
)
include_directories(SYSTEM ${MY_INCLUDE_PATHS} SYSTEM ${TORCH_INCLUDE_DIRS})

set(GMS_LIB_HEADERS
    gmslib/argparser.hpp
    gmslib/base.hpp
    gmslib/gaussian.hpp
    gmslib/geom.hpp
    gmslib/io.hpp
    gmslib/memoryinfo.hpp
    gmslib/mixture.hpp
    gmslib/parallel.hpp
    gmslib/pointindex.hpp
    gmslib/pointset.hpp
    gmslib/random.hpp
    gmslib/sphereindex.hpp
    gmslib/timer.hpp
    gmslib/vec.hpp
)
set(COMPUTE_MIXTURE_SOURCES
    computeMixture.cpp
)

add_executable(computeMixture ${GMS_LIB_HEADERS} ${COMPUTE_MIXTURE_SOURCES})
target_link_libraries(computeMixture PUBLIC OpenMP::OpenMP_CXX)

set(PYTHON_BINDINGS_SOURCES
    pytorch_bindings/gms_compute_mixture_binding.cpp
)
add_library(python_bindings ${PYTHON_BINDINGS_SOURCES} ${GMS_LIB_HEADERS})
target_link_libraries(python_bindings PUBLIC OpenMP::OpenMP_CXX torch)
