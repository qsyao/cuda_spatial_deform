#include "ops_copy.cuh"

__global__ void device_apply_scale(float* coords, float scale, size_t total_size){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        i < total_size;
        i += blockDim.x * gridDim.x){
        coords[i] = coords[i] * scale;
    }
    __syncthreads();
}

__global__ void recenter_2D(float* coords, size_t dim_y, size_t dim_x){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < dim_x * dim_y){
        coords[index] += (float)dim_y/2.0;
        coords[index + dim_x*dim_y] += (float)dim_x/2.0;
    }
    __syncthreads();
}

__global__ void recenter_3D(float* coords, size_t dim_z, size_t dim_y, size_t dim_x){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim_x * dim_y * dim_z;
    if(index < total){
        coords[index] += (float)dim_z/2.0;
        coords[index + total] += (float)dim_y/2.0;
        coords[index + 2 * total] += (float)dim_x/2.0;
    }
    __syncthreads();
}
