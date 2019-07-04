#ifndef CUDA_AUGMENTATION_UTILS
#define CUDA_AUGMENTATION_UTILS

#include <iostream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

extern "C"{

class Handle {
public:
    Handle() : batchsize(1), dim_x(1), dim_y(1), dim_z(1){}

    void set_2D(size_t y, size_t x);

    void set_3D(size_t z, size_t y, size_t x);

    void copy_input(float* input);

    void do_nothing();

    void copy_output(float* ret);

    void check_coords(float* coords);
    
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
    float* img;
    float* output;
    float* pin_img;
    float* pin_output;

    float* coords;
};
    
}

#endif
