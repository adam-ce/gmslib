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
        self.verbose = True
        self.memory = False
        self.alpha = 2.0
        self.blocksize = 0
        self.pointpos = True
        self.stdev = 0.01
        self.iso = False
        self.inittype = "fixed"
        self.knn = 8
        self.fixeddist = 0.1
        self.weighted = False
        self.levels = 20
        self.threads = 8


def compute_mixture(point_cloud: torch.Tensor, params: Params) -> torch.Tensor:
    par = bindings.ExecutionParams()
    par.verbose = params.verbose
    par.memory = params.memory
    par.alpha = params.alpha
    par.blocksize = params.blocksize
    par.pointpos = params.pointpos
    par.stdev = params.stdev
    par.iso = params.iso
    par.inittype = params.inittype
    par.knn = params.knn
    par.fixeddist = params.fixeddist
    par.weighted = params.weighted
    par.levels = params.levels
    par.threads = params.threads
    return bindings.compute_mixture(point_cloud, par)

