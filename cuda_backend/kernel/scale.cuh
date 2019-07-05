#ifndef CUDA_AUGMENTATION_SCALE
#define CUDA_AUGMENTATION_SCALE

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

__global__ void device_apply_scale(float* coords, float scale, size_t total_size);

#endif