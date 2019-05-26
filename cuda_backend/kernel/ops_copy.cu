#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "ops_copy.cuh"

__global__ void device_only_copy(float* output, float* input, size_t total_size){
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            i < total_size;
            i += blockDim.x * gridDim.x){
        output[i] = input[i];
    }
    __syncthreads();
}

void only_copy(float* output, float* input, size_t total_size){
    dim3 threads(min(total_size, (long)512), 1, 1);
    dim3 blocks(total_size/512 + 1, 1, 1);
    device_only_copy<<<blocks, threads>>>(output, input, total_size);
}
