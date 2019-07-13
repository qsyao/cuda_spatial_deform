#include "utils.cuh"

#include "ops_copy.cuh"
#include "spatial_deform.cuh"
#include "interpolate.cuh"

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
    std::cout<<"Malloc "<< 6 * total_size * sizeof(float)/1024/1024
             << "MB"<<std::endl;

    checkCudaErrors(cudaMalloc((void **)&img,
                            total_size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&output,
                            total_size * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&pin_img,
                            total_size * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&pin_output,
                            total_size * sizeof(float)));
   
    checkCudaErrors(cudaMalloc((void **)&random,
                            coords_size * sizeof(float)));     

    checkCudaErrors(cudaMalloc((void **)&coords,
                        coords_size * sizeof(float)));    
    checkCudaErrors(cudaMallocHost((void **)&pin_coords,
                        coords_size * sizeof(float)));
    
    dim3 threads(min(total_size, (long)512), 1, 1);
    dim3 blocks(total_size/512 + 1, 1, 1);
    set_coords_2D<<<blocks, threads, 0, stream>>>(coords, dim_y, dim_x);
    checkCudaErrors(cudaStreamSynchronize(stream));
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

    std::cout<<"Malloc "<< 8 * total_size * sizeof(float)/1024/1024
             << "MB"<<std::endl;

    checkCudaErrors(cudaMalloc((void **)&gpu_rot_matrix, 9 * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&img,
                            total_size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&output,
                            total_size * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&pin_img,
                            total_size * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&pin_output,
                            total_size * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&random,
                            coords_size * sizeof(float)));      

    checkCudaErrors(cudaMalloc((void **)&coords,
                        coords_size * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&pin_coords,
                        coords_size * sizeof(float)));
 
    dim3 threads(min(total_size, (long)512), 1, 1);
    dim3 blocks(total_size/512 + 1, 1, 1);
    set_coords_3D<<<blocks, threads, 0, stream>>>(coords, dim_z, dim_y, dim_x);
    checkCudaErrors(cudaStreamSynchronize(stream));
}

void Handle::scale(float scale){
    assert(scale <= 1.0 && scale > 0.0);
    dim3 threads(min(coords_size, (long)512), 1, 1);
    dim3 blocks(coords_size/512 + 1, 1, 1);
    device_apply_scale<<<blocks, threads, 0, stream>>>(coords, scale, coords_size);
}

void Handle::flip(int do_x, int do_y, int do_z){
    if(is_3D){
        dim3 threads(min(total_size, (long)512), 1, 1);
        dim3 blocks(total_size/512 + 1, 1, 1);
        flip_3D<<<blocks, threads, 0, stream>>>(coords, dim_z, dim_y, dim_x,
                                                do_z, do_y, do_x);
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
    else{
        dim3 threads(min(total_size, (long)512), 1, 1);
        dim3 blocks(total_size/512 + 1, 1, 1);
        flip_2D<<<blocks, threads, 0, stream>>>(coords, dim_y, dim_x, do_y, do_x);
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
}

void Handle::host_rotate_2D(float angle){
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);
    dim3 threads(min(total_size, (long)512), 1, 1);
    dim3 blocks(total_size/512 + 1, 1, 1);
    rotate_2D<<<blocks, threads, 0, stream>>>(coords, dim_y, dim_x, cos_angle, sin_angle);
    checkCudaErrors(cudaStreamSynchronize(stream));    
}

void Handle::host_rotate_3D(float* rot_matrix){
    checkCudaErrors(cudaMemcpyAsync(gpu_rot_matrix, 
                                    rot_matrix, 
                                    9 * sizeof(float),
                                    cudaMemcpyHostToDevice, 
                                    stream));
    dim3 threads(min(total_size, (long)512), 1, 1);
    dim3 blocks(total_size/512 + 1, 1, 1);
    rotate_3D<<<blocks, threads, 0, stream>>>(coords, dim_z, dim_y, dim_x, gpu_rot_matrix);
    checkCudaErrors(cudaStreamSynchronize(stream));    
}

void Handle::elastic(float sigma, float alpha, float truncate,
                                int mode_type, float c_val){
    // generate random offset by coords.size
    checkCudaErrors(curandGenerateUniform(gen, random, coords_size));
    // make the radius of the filter equal to truncate standard deviations
    int lw = int(sigma * truncate + 0.5);
    float sigma2 = sigma * sigma;
    // generate kernel
    float total = 0;
    for(int i = -lw; i < lw + 1; i++){
        kernel_pin[i + lw] = exp(i * i * -0.5 / sigma2);
        total += kernel_pin[i + lw];
    }
    for(int i = -lw; i < lw + 1; i++){
        kernel_pin[i + lw] = kernel_pin[i + lw] / total;
    }

    // Copy kernel
    checkCudaErrors(cudaMemcpyAsync(kernel, 
                                    kernel_pin, 
                                    (2 * lw + 1) * sizeof(float),
                                    cudaMemcpyHostToDevice,
                                    stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    
    if(is_3D){
        dim3 threads(min(coords_size, (long)512), 1, 1);
        dim3 blocks(coords_size/512 + 1, 1, 1);
        scale_random<<<blocks, threads, 0, stream>>>(random, coords_size);

        gussain_filter_x<<<blocks, threads, 0, stream>>>(random, kernel, lw, dim_z,
                                                    dim_y, dim_x, mode_type, c_val);
        gussain_filter_y<<<blocks, threads, 0, stream>>>(random, kernel, lw, dim_z,
                                                    dim_y, dim_x, mode_type, c_val);
        gussain_filter_z<<<blocks, threads, 0, stream>>>(random, kernel, lw, dim_z,
                                                    dim_y, dim_x, mode_type, c_val);

        plus_offsets<<<blocks, threads, 0, stream>>>(coords, random, coords_size, alpha);
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
    else{
        dim3 threads(min(coords_size, (long)512), 1, 1);
        dim3 blocks(coords_size/512 + 1, 1, 1);
        scale_random<<<blocks, threads, 0, stream>>>(random, coords_size);

        gussain_filter_x<<<blocks, threads, 0, stream>>>(random, kernel, lw, 1,
                                                    dim_y, dim_x, mode_type, c_val);
        gussain_filter_y<<<blocks, threads, 0, stream>>>(random, kernel, lw, 1,
                                                    dim_y, dim_x, mode_type, c_val);

        plus_offsets<<<blocks, threads, 0, stream>>>(coords, random, coords_size, alpha);
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
}

void Handle::translate(float seg_x, float seg_y, float seg_z){
    if(is_3D){
        dim3 threads(min(total_size, (long)512), 1, 1);
        dim3 blocks(total_size/512 + 1, 1, 1);
        translate_3D<<<blocks, threads, 0, stream>>>(coords, dim_z, dim_y, dim_x,
                                                seg_z, seg_y, seg_x);
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
    else{
        dim3 threads(min(total_size, (long)512), 1, 1);
        dim3 blocks(total_size/512 + 1, 1, 1);
        translate_2D<<<blocks, threads, 0, stream>>>(coords, dim_y, dim_x, seg_y, seg_x);
        checkCudaErrors(cudaStreamSynchronize(stream));
    }    
}

void Handle::copy_input(float* input){
    memcpy(pin_img, input, total_size * sizeof(float));
    checkCudaErrors(cudaMemcpyAsync(img, pin_img, total_size * sizeof(float),
                            cudaMemcpyHostToDevice, stream));
}

void Handle::do_nothing(){
    only_copy(output, img, total_size);
}

void Handle::copy_output(float* ret){
    checkCudaErrors(cudaMemcpyAsync(pin_output, output, total_size * sizeof(float),
                            cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    memcpy(ret, pin_output, total_size * sizeof(float));
}

void Handle::check_coords(float* output){
    checkCudaErrors(cudaMemcpyAsync(pin_coords, coords, coords_size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    memcpy(output, pin_coords, coords_size * sizeof(float));       
}

void Handle::interpolate(int order){
    dim3 threads(min(total_size, (long)512), 1, 1);
    dim3 blocks(total_size/512 + 1, 1, 1);

    if(is_3D){
        interplate_3D<<<blocks, threads, 0, stream>>>(coords, img, output, order,
                                                            dim_z, dim_y, dim_x, mode_type, c_val);
    }
    else{
        interplate_2D<<<blocks, threads, 0, stream>>>(coords, img, output, order,
                                                             dim_y, dim_x, mode_type, c_val);
    }
}

void Handle::reset(){
    if(is_3D){
        dim3 threads(min(total_size, (long)512), 1, 1);
        dim3 blocks(total_size/512 + 1, 1, 1);
        set_coords_3D<<<blocks, threads, 0, stream>>>(coords, dim_z, dim_y, dim_x);
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
    else{
        dim3 threads(min(total_size, (long)512), 1, 1);
        dim3 blocks(total_size/512 + 1, 1, 1);
        set_coords_2D<<<blocks, threads, 0, stream>>>(coords, dim_y, dim_x);
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
}

void Handle::recenter(){
    if(is_3D){
        dim3 threads(min(total_size, (long)512), 1, 1);
        dim3 blocks(total_size/512 + 1, 1, 1);
        recenter_3D<<<blocks, threads, 0, stream>>>(coords, dim_z, dim_y, dim_x);
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
    else{
        dim3 threads(min(total_size, (long)512), 1, 1);
        dim3 blocks(total_size/512 + 1, 1, 1);
        recenter_2D<<<blocks, threads, 0, stream>>>(coords, dim_y, dim_x);
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
}
