#include "utils.cuh"

extern "C" {

Handle* init_2D_handle(size_t y, size_t x){
    Handle *ret = new Handle();
    ret->set_2D(y, x);
    return ret;
}

Handle* init_3D_handle(size_t z, size_t y, size_t x){
    Handle *ret = new Handle();
    ret->set_3D(z, y, x);
    return ret;
}

void test(Handle* cuda_handle, float* output, float* input){
    cuda_handle->copy_input(input);
    cuda_handle->do_nothing();
    cuda_handle->copy_output(output);
}

void check_coords(Handle* cuda_handle, float* coords){
    cuda_handle->check_coords(coords);
}

} // extern "C"
