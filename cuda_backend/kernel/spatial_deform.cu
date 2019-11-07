#include "spatial_deform.cuh"

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
    size_t id_z = index / (dim_x * dim_y);
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

__global__ void translate_3D(float* coords,
                            size_t dim_z,
                            size_t dim_y, 
                            size_t dim_x,
                            float seg_z,
                            float seg_y,
                            float seg_x){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim_x * dim_y * dim_z;
    if(index < total){
        coords[index] += seg_z;
        coords[index + total] += seg_y;
        coords[index + total * 2] += seg_x;
        __syncthreads();
    }
}

__global__ void translate_2D(float* coords, 
                            size_t dim_y, 
                            size_t dim_x,
                            float seg_y,
                            float seg_x){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim_x * dim_y;
    if(index < total){
        coords[index] += seg_y;
        coords[index + total] += seg_x;
        __syncthreads();
    }
}

__global__ void rotate_2D(float* coords, 
                        size_t dim_y, 
                        size_t dim_x,
                        float cos_angle,
                        float sin_angle){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim_x * dim_y;
    float new_y, new_x;
    float old_y = coords[index];
    float old_x = coords[index + total];
    if(index < total){
        new_y = cos_angle * old_y + sin_angle * old_x;
        new_x = -sin_angle * old_y + cos_angle * old_x;
        __syncthreads();
        coords[index] = new_y;
        coords[index + total] = new_x;
        __syncthreads();
    }
}

__global__ void rotate_3D(float* coords, 
                        size_t dim_z,
                        size_t dim_y, 
                        size_t dim_x,
                        float* rot_matrix){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim_x * dim_y * dim_z;
    float new_y = 0, new_x = 0, new_z = 0;
    float old_z = coords[index];
    float old_y = coords[index + total];
    float old_x = coords[index + 2 * total];
    if(index < total){
        new_z = old_z * rot_matrix[0] + old_y * rot_matrix[3] + old_x * rot_matrix[6];
        new_y = old_z * rot_matrix[1] + old_y * rot_matrix[4] + old_x * rot_matrix[7];
        new_x = old_z * rot_matrix[2] + old_y * rot_matrix[5] + old_x * rot_matrix[8];
        __syncthreads();
        coords[index] = new_z;
        coords[index + total] = new_y;
        coords[index + 2 * total] = new_x;
        __syncthreads();
    }
}

__device__ __forceinline__ int mirror(int index, int len){
    int s2 = 2 * len - 2;
    if(index < 0){
        index = s2 * (-index / s2) + index;
        return index <= 1 - len ? index + s2 : -index;
    }
    if(index >= len){
        index -= s2 * (index / s2);
        if(index >= len) 
            index = s2 - index;
        return index;
    }
    if(index < 0 || index >= len) index = mirror(index, len);
    return index;
}

__global__ void scale_random(float *random, size_t total_size){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < total_size){
        random[index] = random[index] * 2.0 - 1.0;
        __syncthreads();
    }
}

__global__ void plus_offsets(float *coords, float *random, size_t total_size, float alpha){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < total_size){
        coords[index] += random[index] * alpha;
        __syncthreads();
    }
}

__global__ void gussain_filter_x(float* random,
                                float* kernel, 
                                int lw,
                                size_t dim_z,
                                size_t dim_y,
                                size_t dim_x,
                                int mode,
                                float cval){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim_x * dim_y * dim_z;
    size_t total_xy = dim_x * dim_y;
    size_t id_x = index % dim_x;
    size_t id_y = (index / dim_x) % dim_y;
    size_t id_z = (index / total_xy) % dim_z;
    size_t id_block = index / total;
    int id;
    float new_pixel = 0;
    int dim = 2;
    if(dim_z > 1){
        dim = 3;
    }
    if(index < total * dim){
        if(mode == 0){
            for(int i = -lw; i < lw + 1; i++){
                id = id_x + i;
                if(id < 0 || id > dim_x - 1)
                    new_pixel += cval * kernel[i+lw];
                else new_pixel += kernel[i+lw] * 
                        random[id_block * total + id_z * total_xy + id_y * dim_x + id];
            }
            __syncthreads();
            random[index] = new_pixel;
            __syncthreads();
        }
        else{
            for(int i = -lw; i < lw + 1; i++){
                id = id_x + i;
                id = mirror(id, dim_x);
                new_pixel += kernel[i+lw] * 
                       random[id_block * total + id_z * total_xy + id_y * dim_x + id];
            }
            __syncthreads();
            random[index] = new_pixel;
            __syncthreads();   
        }
    }
}

__global__ void gussain_filter_y(float* random,
                                float* kernel, 
                                int lw,
                                size_t dim_z,
                                size_t dim_y,
                                size_t dim_x,
                                int mode,
                                float cval){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim_x * dim_y * dim_z;
    size_t total_xy = dim_x * dim_y;
    size_t id_x = index % dim_x;
    size_t id_y = (index / dim_x) % dim_y;
    size_t id_z = (index / total_xy) % dim_z;
    size_t id_block = index / total;
    int id;
    float new_pixel = 0;
    int dim = 2;
    if(dim_z > 1){
        dim = 3;
    }
    if(index < total * dim){
        if(mode == 0){
            for(int i = -lw; i < lw + 1; i++){
                id = id_y + i;
                if(id < 0 || id > dim_y - 1)
                    new_pixel += cval * kernel[i+lw];
                else new_pixel += kernel[i+lw] * 
                        random[id_block * total + id_z * total_xy + id * dim_x + id_x];
            }
            __syncthreads();
            random[index] = new_pixel;
            __syncthreads();
        }
        else{
            for(int i = -lw; i < lw + 1; i++){
                id = id_y + i;
                id = mirror(id, id_y);
                new_pixel += kernel[i+lw] * 
                       random[id_block * total + id_z * total_xy + id * dim_x + id_x];
            }
            __syncthreads();
            random[index] = new_pixel;
            __syncthreads();   
        }
    }
}

__global__ void gussain_filter_z(float* random,
                                float* kernel, 
                                int lw,
                                size_t dim_z,
                                size_t dim_y,
                                size_t dim_x,
                                int mode,
                                float cval){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim_x * dim_y * dim_z;
    size_t total_xy = dim_x * dim_y;
    size_t id_x = index % dim_x;
    size_t id_y = (index / dim_x) % dim_y;
    size_t id_z = (index / total_xy) % dim_z;
    size_t id_block = index / total;
    int id;
    float new_pixel = 0;
    int dim = 2;
    if(dim_z > 1){
        dim = 3;
    }
    if(index < total * dim){
        if(mode == 0){
            for(int i = -lw; i < lw + 1; i++){
                id = id_z + i;
                if(id < 0 || id > dim_z - 1)
                    new_pixel += cval * kernel[i+lw];
                else new_pixel += kernel[i+lw] * 
                        random[id_block * total + id * total_xy + id_y * dim_x + id_x];
            }
            __syncthreads();
            random[index] = new_pixel;
            __syncthreads();
        }
        else{
            for(int i = -lw; i < lw + 1; i++){
                id = id_z + i;
                id = mirror(id, id_z);
                new_pixel += kernel[i+lw] * 
                       random[id_block * total + id * total_xy + id_y * dim_x + id_x];
            }
            __syncthreads();
            random[index] = new_pixel;
            __syncthreads();   
        }
    }
}

