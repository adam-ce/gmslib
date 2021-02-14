//-----------------------------------------------------------------------------
// gmslib - Gaussian Mixture Surface Library
// Copyright (c) Adam Celarek 2021 <celarek at cg dot tuwien dot ac dot at>
// 
// Usage is subject to the terms of the WFP (modified BSD-3-Clause) license.
// See the accompanied LICENSE file or
// https://github.com/rpreiner/gmslib/blob/main/LICENSE
//-----------------------------------------------------------------------------

#include <cassert>

#include <omp.h>
#include <torch/extension.h>

#include "gmslib/pointset.hpp"
#include "gmslib/mixture.hpp"
using namespace std;
using namespace gms;


// todo: find out how to forward a struct or similar from python side, or use a lengthy list of parameters and fill params.
// todo: add a verbosity parameter to kill all stdout

torch::Tensor compute_mixture(torch::Tensor point_cloud) {

    Mixture::Params params;
    // be careful with omp_get_num_procs.
    // starting 24 threads takes longer than processing 10k uniformly distributed points. best make it a parameter as well.
    // might be different when using real data.
//    params.numThreads = omp_get_num_procs();
    point_cloud = point_cloud.clone().contiguous().toType(torch::ScalarType::Float).cpu();
    TORCH_CHECK(point_cloud.sizes().size() == 2, "point cloud must have dimensions of N x 3")
    TORCH_CHECK(point_cloud.size(0) > 0, "point cloud must have dimensions of N x 3")
    TORCH_CHECK(point_cloud.size(1) == 3, "point cloud must have dimensions of N x 3")

    auto n = point_cloud.size(0);
    PointSet cpp_point_cloud;
    cpp_point_cloud.reserve(size_t(n));
    auto point_cloud_accessor = point_cloud.accessor<float, 2>();
    for (unsigned i = 0; i < n; ++i) {
        cpp_point_cloud.emplace_back(point_cloud_accessor[i][0], point_cloud_accessor[i][1], point_cloud_accessor[i][2]);
    }

    BBox bboxPoints(cpp_point_cloud);
    float conversionFac = 0.01f * length(bboxPoints.dim());
    params.maxInitNeighborDist *= conversionFac;
    params.initIsotropicStdev *= conversionFac;

    Mixture cpp_mixture(&cpp_point_cloud, params);

    auto torch_mixture = torch::empty({int(cpp_mixture.size()), 13}, torch::ScalarType::Float);
    auto mixture_accessor = torch_mixture.accessor<float, 2>();
    for (unsigned i = 0; i < cpp_mixture.size(); ++i) {
        mixture_accessor[i][0] = cpp_mixture[i].weight;

        mixture_accessor[i][1] = cpp_mixture[i].mu.x;
        mixture_accessor[i][2] = cpp_mixture[i].mu.y;
        mixture_accessor[i][3] = cpp_mixture[i].mu.z;

        mixture_accessor[i][4] = cpp_mixture[i].cov.e00;
        mixture_accessor[i][5] = cpp_mixture[i].cov.e01;
        mixture_accessor[i][6] = cpp_mixture[i].cov.e02;

        mixture_accessor[i][7] = cpp_mixture[i].cov.e01;
        mixture_accessor[i][8] = cpp_mixture[i].cov.e11;
        mixture_accessor[i][9] = cpp_mixture[i].cov.e12;

        mixture_accessor[i][10] = cpp_mixture[i].cov.e02;
        mixture_accessor[i][11] = cpp_mixture[i].cov.e12;
        mixture_accessor[i][12] = cpp_mixture[i].cov.e22;
    }

    return torch_mixture;
}


#ifndef GMSLIB_CMAKE_TEST_BUILD
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_mixture", &compute_mixture, "Compute mixture using HEM algorithm");
}
#endif
