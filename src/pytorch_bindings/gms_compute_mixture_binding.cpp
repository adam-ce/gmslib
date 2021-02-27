//-----------------------------------------------------------------------------
// gmslib - Gaussian Mixture Surface Library
// Copyright (c) Adam Celarek 2021 <celarek at cg dot tuwien dot ac dot at>
//               Simon Fraiss 2021
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

struct ExecutionParams {
    bool verbose = true;        //Verbose output
    bool memory = false;        //Activate memory usage profiling
    float alpha = 2.0;          //Clustering regularization parameter
    unsigned int blocksize = 0; //Compute mixture in blocks of the specified point count
    bool pointpos = true;       //Initializes Gaussian positions in point locations rather than local point means
    float stdev = 0.01f;        //Default isotropic standard deviation bias of each initial Gaussian [in %bbd]
    bool iso = false;           //Initialize mixture with isotropic Gaussians of standard deviation <stdev>
    string inittype = "fixed";  //'knn' - Init anisotropic Gaussians based on KNN; 'fixed' - based on fixed distance
    unsigned int knn = 8;       //Number of nearest neighbors considered for 'knn' initialization
    float fixeddist = 0.1f;     //Max neighborhood distance for points considered for 'fixed' initialization [in %bbd]
    bool weighted = false;      //Initializes mixture with locally normalized density
    unsigned int levels = 20;   //Number of HEM clustering levels
    unsigned int threads = 8;   //Number of parallel threads
    unsigned int ngaussians = 0;//Fixed Number of output Gaussians if desired, otherwise zero. If not zero, levels will be ignored, and chosen automatically.
    float reductionFactor = 3;  //Reduction Factor
    //Quantities described with '[in %bbd]' are given in percent of the input point cloud bounding box diagonal.
};

// todo: add a verbosity parameter to kill all stdout

torch::Tensor compute_mixture(torch::Tensor point_cloud, ExecutionParams execparams) {

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

    Mixture::Params params;
    params.computeNVar = false;		// deactivate CLOP normal clustering 
    params.verbose = execparams.verbose;
    params.memoryProfiling = execparams.memory;
    params.alpha = execparams.alpha;
    params.blockSize = execparams.blocksize;
    params.blockProcessing = execparams.blocksize > 0;
    params.hemReductionFactor = execparams.reductionFactor;
    params.initIsotropic = execparams.iso;
    params.initIsotropicStdev = execparams.stdev;
    params.initMeansInPoints = execparams.pointpos;
    params.kNNCount = execparams.knn;
    params.maxInitNeighborDist = execparams.fixeddist;
    params.nLevels = execparams.levels;
    params.numThreads = execparams.threads;
    params.useWeightedPotentials = execparams.weighted;
    params.initNeighborhoodType = 0;
    params.fixedNumberOfGaussians = execparams.ngaussians;
    if (execparams.inittype != "")
    {
        if (execparams.inittype == "fixed")		params.initNeighborhoodType = 0;
        else if (execparams.inittype == "knn")	params.initNeighborhoodType = 1;
        else {
            std::cerr << "Invalid 'anisotype' argument. Use 'knn' or 'fixed'.\n";
            exit(1);
        }
    }

    BBox bboxPoints(cpp_point_cloud);
    float conversionFac = 0.01f * length(bboxPoints.dim());
    params.maxInitNeighborDist *= conversionFac;
    params.initIsotropicStdev *= conversionFac;

    Mixture cpp_mixture(&cpp_point_cloud, params);

    auto torch_mixture = torch::empty({int(cpp_mixture.size()), 13}, torch::ScalarType::Float);
    auto mixture_accessor = torch_mixture.accessor<float, 2>();
    for (unsigned i = 0; i < cpp_mixture.size(); ++i) {
        mixture_accessor[i][0] = cpp_mixture[i].weight / n;

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

    py::class_<ExecutionParams>(m, "ExecutionParams")
        .def(py::init<>())
        .def_readwrite("verbose", &ExecutionParams::verbose)
        .def_readwrite("memory", &ExecutionParams::memory)
        .def_readwrite("alpha", &ExecutionParams::alpha)
        .def_readwrite("blocksize", &ExecutionParams::blocksize)
        .def_readwrite("pointpos", &ExecutionParams::pointpos)
        .def_readwrite("stdev", &ExecutionParams::stdev)
        .def_readwrite("iso", &ExecutionParams::iso)
        .def_readwrite("inittype", &ExecutionParams::inittype)
        .def_readwrite("knn", &ExecutionParams::knn)
        .def_readwrite("fixeddist", &ExecutionParams::fixeddist)
        .def_readwrite("weighted", &ExecutionParams::weighted)
        .def_readwrite("levels", &ExecutionParams::levels)
        .def_readwrite("reductionFactor", &ExecutionParams::reductionFactor)
        .def_readwrite("threads", &ExecutionParams::threads)
        .def_readwrite("ngaussians", &ExecutionParams::ngaussians);

    m.def("compute_mixture", &compute_mixture, "Compute mixture using HEM algorithm");
}
#endif
