#include "utils.cuh"
#include "ops_copy.cuh"
#include "scale.cuh"

__global__ void set_coords_2D(float* coords, size_t y, size_t x){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t id_x = index % x;
    size_t id_y = index / x;
    if(index < x * y){
        coords[id_x + id_y * x] = id_y - (float)y/2.0;
        coords[id_x + id_y * x + x*y] = id_x - (float)x/2.0;
    }
    __syncthreads();
}

__global__ void set_coords_3D(float* coords, size_t z, size_t y, size_t x){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t id_x = index % x;
    size_t id_y = (index / x) % y;
    size_t id_z = index / (x * y);
    if(index < x * y * z){
        coords[index] = id_z - (float)z/2.0;
        coords[index + x * y * z] = id_y - (float)y/2.0;
        coords[index + 2 * x * y * z] = id_x -(float)x/2.0;
    }
    __syncthreads();
}

void Handle::set_2D(size_t y, size_t x){
    is_3D = false;
    dim_x = x;
    dim_y = y;
    total_size = dim_x * dim_y;
    coords_size = total_size * 2;

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

    checkCudaErrors(cudaMalloc((void **)&coords,
                        coords_size * sizeof(float)));    
    checkCudaErrors(cudaMallocHost((void **)&pin_coords,
                        coords_size * sizeof(float)));
    
    dim3 threads(min(total_size, (long)512), 1, 1);
    dim3 blocks(total_size/512 + 1, 1, 1);
    set_coords_2D<<<blocks, threads>>>(coords, dim_y, dim_x);
    checkCudaErrors(cudaDeviceSynchronize());
}

void Handle::scale(float scale){
    host_apply_scale(coords, scale, coords_size);
}

void Handle::set_3D(size_t z, size_t y, size_t x){
    is_3D = true;
    dim_x = x;
    dim_y = y;
    dim_z = z;
    total_size = dim_x * dim_y * dim_z;
    coords_size = total_size * 3;

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

    checkCudaErrors(cudaMalloc((void **)&coords,
                        coords_size * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&pin_coords,
                        coords_size * sizeof(float)));

    dim3 threads(min(total_size, (long)512), 1, 1);
    dim3 blocks(total_size/512 + 1, 1, 1);
    set_coords_3D<<<blocks, threads>>>(coords, dim_z, dim_y, dim_x);
    checkCudaErrors(cudaDeviceSynchronize());
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
    checkCudaErrors(cudaDeviceSynchronize());
    memcpy(ret, pin_output, total_size * sizeof(float));
}

void Handle::check_coords(float* output){
    checkCudaErrors(cudaMemcpyAsync(pin_coords, coords, coords_size * sizeof(float),
                        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    memcpy(output, pin_coords, coords_size * sizeof(float));       
}

