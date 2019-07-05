#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "ops_copy.cuh"

__global__ void device_apply_scale(float* coords, float scale, size_t total_size){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        i < total_size;
        i += blockDim.x * gridDim.x){
        coords[i] = coords[i] * scale;
    }
    __syncthreads();
}

void host_apply_scale(float* coords, float scale, size_t total_size){
    assert(scale <= 1.0 && scale > 0.0);
    dim3 threads(min(total_size, (long)512), 1, 1);
    dim3 blocks(total_size/512 + 1, 1, 1);
    device_apply_scale<<<blocks, threads>>>(coords, scale, total_size);
}
