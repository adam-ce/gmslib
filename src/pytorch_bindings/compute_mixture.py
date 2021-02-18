import os
import platform

import torch
import torch.utils.cpp_extension

source_dir = os.path.dirname(__file__)
# print(source_dir)

extra_include_paths = [source_dir + "/..", source_dir + "/../ext"]
if platform.system() == "Windows":
    cpp_extra_cflags = ["/openmp", "/O2", "/std:c++17", "/DNDEBUG", "/D_HAS_STD_BYTE=0", "/DNOMINMAX"]
else:
    cpp_extra_cflags = ["-fopenmp", "-ffast-math", " -fno-finite-math-only", "-O4", "-march=native", "--std=c++17", "-DNDEBUG"]

bindings = torch.utils.cpp_extension.load('compute_mixture',
                [source_dir + '/gms_compute_mixture_binding.cpp'],
                extra_include_paths=extra_include_paths, verbose=True, extra_cflags=cpp_extra_cflags, extra_ldflags=["-lpthread"])


class Params:
    def __init__(self):
        self.verbose = False
        self.memoryProfiling = False
        self.initNeighborhoodType = 1
        self.kNNCount = 8
        self.maxInitNeighborDist = 1.0
        self.initIsotropicStdev = 1.0
        self.initIsotropic = False
        self.useWeightedPotentials = True
        self.initMeansInPoints = True
        self.nLevels = 4
        self.hemReductionFactor = 3.0
        self.alpha = 2.2
        self.fixedNumberOfGaussians = 0
        self.computeNVar = True
        self.blockProcessing = False
        self.blockSize = 1000000
        self.numThreads = 8


def compute_mixture(point_cloud: torch.Tensor, params: Params) -> torch.Tensor:
    par = bindings.Params()
    par.verbose = params.verbose
    par.memoryProfiling = params.memoryProfiling
    par.initNeighborhoodType = params.initNeighborhoodType
    par.kNNCount = params.kNNCount
    par.maxInitNeighborDist = params.maxInitNeighborDist
    par.initIsotropicStdev = params.initIsotropicStdev
    par.initIsotropic = params.initIsotropic
    par.useWeightedPotentials = params.useWeightedPotentials
    par.initMeansInPoints = params.initMeansInPoints
    par.nLevels = params.nLevels
    par.hemReductionFactor = params.hemReductionFactor
    par.alpha = params.alpha
    par.fixedNumberOfGaussians = params.fixedNumberOfGaussians
    par.computeNVar = params.computeNVar
    par.blockProcessing = params.blockProcessing
    par.blockSize = params.blockSize
    par.numThreads = params.numThreads
    return bindings.compute_mixture(point_cloud, par)

