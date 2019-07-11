#ifndef CUDA_AUGMENTATION_MODE
#define CUDA_AUGMENTATION_MODE

#include <cuda.h>
#include "curand.h"
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

__device__ float lookup_first(float index, int max_dim, int mode);

__device__ float lookup_second(float index, int max_dim);

#endif
