#include "utils.cuh"
#include "ops_copy.cuh"

void Handle::set_2D(size_t y, size_t x){
    dim_x = x;
    dim_y = y;
    total_size = dim_x * dim_y;

    std::cout<<"Malloc for 2D image ----------\n"
             <<" dim_x : "<<dim_x
             <<" dim_y : "<<dim_y
             <<" total : "<<total_size<<std::endl;
    std::cout<<"Malloc "<< total_size * sizeof(float)/1024/1024
             << "MB(double)"<<std::endl;

    checkCudaErrors(cudaMalloc((void **)&img,
                            total_size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&output,
                            total_size * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&pin_img,
                            total_size * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&pin_output,
                            total_size * sizeof(float)));
}

void Handle::set_3D(size_t z, size_t y, size_t x){
    dim_x = x;
    dim_y = y;
    dim_z = z;
    total_size = dim_x * dim_y * dim_z;

    std::cout<<"Malloc for 3D image ----------\n"
             <<" dim_x : "<<dim_x
             <<" dim_y : "<<dim_y
             <<" dim_z : "<<dim_z
             <<" total : "<<total_size<<std::endl;

    std::cout<<"Malloc "<< total_size * sizeof(float)/1024/1024
             << "MB(double)"<<std::endl;

    checkCudaErrors(cudaMalloc((void **)&img,
                            total_size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&output,
                            total_size * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&pin_img,
                            total_size * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&pin_output,
                            total_size * sizeof(float)));
}

void Handle::copy_input(float* input){
    memcpy(pin_img, input, total_size * sizeof(float));
    checkCudaErrors(cudaMemcpyAsync(img, pin_img, total_size * sizeof(float),
                            cudaMemcpyHostToDevice));
}

void Handle::do_nothing(){
    only_copy(output, img, total_size);
}

void Handle::copy_output(float* ret){
    checkCudaErrors(cudaMemcpyAsync(pin_output, output, total_size * sizeof(float),
                            cudaMemcpyDeviceToHost));
    memcpy(ret, pin_output, total_size * sizeof(float));
    checkCudaErrors(cudaDeviceSynchronize());
}
