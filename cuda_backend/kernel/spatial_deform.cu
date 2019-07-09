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

__device__ void exchange(float &a, float &b){
    float temp = a;
    a = b;
    b = temp;
}

__global__ void flip_2D(float* coords, 
                        size_t dim_y, 
                        size_t dim_x,
                        int do_y,
                        int do_x){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim_x * dim_y;
    size_t id_x = index % dim_x;
    size_t id_y = index / dim_x;
    if(index < total){
        if(do_x && id_x < (dim_x / 2)){
            exchange(coords[total + id_y * dim_x + id_x], 
                     coords[total + id_y * dim_x + dim_x-1 - id_x]);
            __syncthreads();
        }
        if(do_y && id_y < (dim_y / 2)){
            exchange(coords[id_y * dim_x + id_x], coords[(dim_y-1 - id_y) * dim_x + id_x]);
            __syncthreads();
        }
    }
}

__global__ void flip_3D(float* coords,
                        size_t dim_z,
                        size_t dim_y, 
                        size_t dim_x,
                        int do_z,
                        int do_y,
                        int do_x){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim_x * dim_y * dim_z;
    size_t total_xy = dim_x * dim_y;
    size_t id_x = index % dim_x;
    size_t id_y = (index / dim_x) % dim_x;
    size_t id_z = index / (dim_x * dim_x);
    if(index < total){
        if(do_x && id_x < (dim_x / 2)){
            exchange(coords[2 * total + id_z * total_xy + id_y * dim_x + id_x], 
                     coords[2 * total + id_z * total_xy + id_y * dim_x + dim_x-1 - id_x]);
            __syncthreads();
        }
        if(do_y && id_y < (dim_y / 2)){
            exchange(coords[total + id_z * total_xy + id_y * dim_x + id_x], 
                     coords[total + id_z * total_xy + (dim_y-1 - id_y) * dim_x + id_x]);
            __syncthreads();
        }
        if(do_z && id_z < (dim_z / 2)){
            exchange(coords[id_z * total_xy + id_y * dim_x + id_x], 
                     coords[(dim_z-1 -id_z) * total_xy + id_y * dim_x + id_x]);
            __syncthreads();
        }
    }
}
