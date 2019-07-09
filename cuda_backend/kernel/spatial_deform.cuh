#ifndef CUDA_AUGMENTATION_SCALE
#define CUDA_AUGMENTATION_SCALE

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

__global__ void device_apply_scale(float* coords, float scale, size_t total_size);

__global__ void recenter_2D(float* coords, size_t dim_y, size_t dim_x);

__global__ void recenter_3D(float* coords, size_t dim_z, size_t dim_y, size_t dim_x);

__global__ void flip_2D(float* coords, 
                        size_t dim_y, 
                        size_t dim_x,
                        int do_y,
                        int do_x);

__global__ void flip_3D(float* coords,
                        size_t dim_z,
                        size_t dim_y, 
                        size_t dim_x,
                        int do_z,
                        int do_y,
                        int do_x);                        

#endif