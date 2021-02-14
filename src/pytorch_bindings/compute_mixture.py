import os
import platform

import torch
import torch.utils.cpp_extension

source_dir = os.path.dirname(__file__)
# print(source_dir)

extra_include_paths = [source_dir + "/..", source_dir + "/../ext"]
if platform.system() == "Windows":
    cpp_extra_cflags = ["/openmp", "/O2", "/std:c++17", "/DNDEBUG"]
else:
    cpp_extra_cflags = ["-fopenmp", "-ffast-math", " -fno-finite-math-only", "-O4", "-march=native", "--std=c++17", "-DNDEBUG"]

bindings = torch.utils.cpp_extension.load('compute_mixture',
                [source_dir + '/gms_compute_mixture_binding.cpp'],
                extra_include_paths=extra_include_paths, verbose=True, extra_cflags=cpp_extra_cflags, extra_ldflags=["-lpthread"])

def compute_mixture(point_cloud: torch.Tensor) -> torch.Tensor:
    bindings.compute_mixture(point_cloud)

