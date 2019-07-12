#ifndef CUDA_AUGMENTATION_UTILS
#define CUDA_AUGMENTATION_UTILS

#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#include <cuda.h>
#include "curand.h"
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

extern "C"{

class Handle {
public:
    Handle(int mode_type, float c_val) : batchsize(1), dim_x(1), dim_y(1),
                 dim_z(1), mode_type(mode_type), c_val(c_val){
        checkCudaErrors(cudaStreamCreate(&stream));
        checkCudaErrors(curandCreateGenerator(&gen, 
                       CURAND_RNG_PSEUDO_DEFAULT));
        checkCudaErrors(cudaMalloc((void **)&kernel,
                            1000 * sizeof(float)));
        checkCudaErrors(cudaMallocHost((void **)&kernel_pin,
                            1000 * sizeof(float)));
        srand(time(NULL));
        int seed = rand();
        checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, seed));
    }

    void set_2D(size_t y, size_t x);

    void set_3D(size_t z, size_t y, size_t x);

    void copy_input(float* input);

    void do_nothing();

    void copy_output(float* ret);

    void check_coords(float* coords);

    void scale(float scale);

    void interpolate_linear();

    void reset();

    void recenter();

    void flip(int do_x, int do_y, int do_z=0);

    void translate(float seg_x=0, float seg_y=0, float seg_z=0);

    void host_rotate_2D(float angle);

    void host_rotate_3D(float* rot_matrix);

    void elastic(float sigma, float alpha, float truncate,
                            int mode_type, float c_valm);

    ~Handle(){
        checkCudaErrors(cudaFree(img));
        checkCudaErrors(cudaFree(output));
        checkCudaErrors(cudaFreeHost(pin_img));
        checkCudaErrors(cudaFreeHost(pin_output));
    }

private:
    bool is_3D = false;
    size_t batchsize;
    size_t dim_z, dim_x, dim_y;
    size_t total_size;
    size_t coords_size;
    float* img;
    float* output;
    float* pin_img;
    float* pin_output;

    float* random;
    float* kernel;
    float* kernel_pin;

    float* gpu_rot_matrix;
    float* coords;
    float* pin_coords;

    int mode_type;
    float c_val;

    cudaStream_t stream;

    curandGenerator_t gen;
};
    
}

#endif
