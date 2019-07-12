#include "utils.cuh"

extern "C" {

Handle* init_2D_handle(size_t y, size_t x, int mode_type, float c_val, int id_gpu){
    Handle *ret = new Handle(mode_type, c_val, id_gpu);
    ret->set_2D(y, x);
    return ret;
}

Handle* init_3D_handle(size_t z, size_t y, size_t x, int mode_type, float c_val, int id_gpu){
    Handle *ret = new Handle(mode_type, c_val, id_gpu);
    ret->set_3D(z, y, x);
    return ret;
}

void linear_interpolate(Handle* cuda_handle, 
                        float* output, 
                        float* input, 
                        int do_reset){
    cuda_handle->copy_input(input);
    cuda_handle->interpolate_linear();
    cuda_handle->copy_output(output);
    if(do_reset)
        cuda_handle->reset();
}

void check_coords(Handle* cuda_handle, float* coords){
    cuda_handle->check_coords(coords);
}

void cu_scale(Handle* cuda_handle, float scale){
    assert(scale > 0.0 && scale < 1.0);
    cuda_handle->scale(scale);
}

void cu_rotate_3D(Handle* cuda_handle, float* rot_matrix){
    cuda_handle->host_rotate_3D(rot_matrix);
}

void cu_flip(Handle* cuda_handle, int do_x, int do_y, int do_z){
    cuda_handle->flip(do_x, do_y, do_z);
}

void cu_translate(Handle* cuda_handle, float seg_x, float seg_y, float seg_z){
    cuda_handle->translate(seg_x, seg_y, seg_z);
}

void cu_rotate_2D(Handle* cuda_handle, float angle){
    cuda_handle->host_rotate_2D(angle);
}

void endding_flag(Handle* cuda_handle){
    cuda_handle->recenter();
}

void cu_elastic(Handle* cuda_handle, float sigma, float alpha, float truncate,
                                    int mode_type, float c_val){
    cuda_handle->elastic(sigma, alpha, truncate, mode_type, c_val);
}

} // extern "C"
