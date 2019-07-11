#ifndef CUDA_AUGMENTATION_INTERPOLATE
#define CUDA_AUGMENTATION_INTERPOLATE

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

__global__ void linear_interplate_2D(float* coords, 
                                     float* img, 
                                     float* output, 
                                     size_t dim_y,
                                     size_t dim_x,
                                     int mode, float cval);

__global__ void linear_interplate_3D(float* coords, 
                                    float* img, 
                                    float* output,
                                    size_t dim_z,
                                    size_t dim_y,
                                    size_t dim_x,
                                    int mode, float cval);
#endif
