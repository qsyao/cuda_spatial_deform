#include "interpolate.cuh"
#include <math.h>

__device__ __forceinline__ float lookup_second(float index, int max_dim){
    int s2 = 2 * (max_dim - 1);
    if (index < 0){
        index = s2 * (int)(-index / s2) + index;
        return index <= 1 - max_dim ? index + s2 : -index;
    }
    if (index >= max_dim - 1){
        index -= s2 * (int)(index / s2);
        return index <= max_dim - 1 ? index : s2 - index;
    }
    return index;
}

__device__ __forceinline__ float lookup_first(float index, int max_dim, int mode){
    /*
        if mode == 'constant':
            type_mode = 0
        elif mode == 'reflect':
            type_mode = 1
        elif mode == 'mirror':
            type_mode = 2
        elif mode == 'nearest':
            type_mode = 3
        elif mode == 'wrap':
            type_mode = 4
        else:
            raise ValueError

        Written occording to Scipy
    */

    if (index >= 0 && index < max_dim - 1){
        return index;
    }
    else{
        if(index < 0){
            if(mode == 1){
                int sz2 = 2 * max_dim;
                if (index < -sz2)
                    index = sz2 * (int)(-index / sz2) + index;
                index = index < -max_dim ? index + sz2 : - index - 1;
                return index;
            }
            if(mode == 2){
                int sz2 = 2 * (max_dim - 1);
                index = sz2 * (int)(-index / sz2) + index;
                return index <= 1 - max_dim ? index + sz2 : -index;
            }
            if(mode == 3){
                return 0;
            }
            if(mode == 4){
                int sz = max_dim - 1;
                return index + sz * ((int)(-index/sz) + 1);
            }
        }
        else{
            if(mode == 1){
                int sz2 = 2 * max_dim;
                index -= sz2 * (int)(index / sz2);
                if (index >= max_dim)
                    index = sz2 - index - 1;
                return index;
            }
            if(mode == 2){
                int sz2 = 2 * max_dim - 2;
                index -= sz2 * (int)(index / sz2);
                if (index >= max_dim)
                    index = sz2 - index;
                return index;
            }
            if(mode == 3){
                return max_dim - 1.00001;
            }
            if(mode == 4){
                int sz = max_dim - 1;
                return index - sz * (int)(index / sz);
            }
        }
    }
}


__global__ void interplate_2D(float* coords, 
                                    float* img, 
                                    float* output,
                                    int order,
                                    size_t dim_y,
                                    size_t dim_x,
                                    int mode, float cval){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < dim_x * dim_y){
        float coords_y = coords[index];
        float coords_x = coords[index + dim_x*dim_y];
        if(mode != 0){
            coords_y = lookup_second(lookup_first(coords_y, (int)dim_y, mode), dim_y);
            coords_x = lookup_second(lookup_first(coords_x, (int)dim_x, mode), dim_x);
        }
        int index_y = (int)floorf(coords_y);
        int index_x = (int)floorf(coords_x);

        float gap_y = coords_y - index_y;
        float gap_x = coords_x - index_x;
        int id_x = index_x, id_x_next = index_x + 1;
        int id_y = index_y, id_y_next = index_y + 1;

        if(mode == 0){
            if (index_y < 0 || index_y >= dim_y -1 ||
                index_x < 0 || index_x >= dim_x -1 ){
                output[index] = cval;
            }
            else{
                if(order > 0)
                    output[index] = gap_x * gap_y * img[dim_x * (index_y + 1) + index_x + 1] +
                                    (1 - gap_x) * gap_y * img[dim_x * (index_y + 1) + index_x] +
                                    gap_x * (1 - gap_y) * img[dim_x * (index_y) + index_x + 1] +
                                    (1 - gap_x) * (1 - gap_y) * img[dim_x * index_y + index_x];
                else{
                    if ( gap_x < 0.5 ) id_x = index_x;
                    else id_x = id_x_next;
                    if ( gap_y < 0.5 ) id_y = index_y;
                    else id_y = id_y_next;
                    output[index] = img[dim_x * id_y + id_x];
                }
            }
        }
        else{
            if(order > 0){
                output[index] = gap_x * gap_y * img[dim_x * id_y_next + id_x_next] +
                                (1 - gap_x) * gap_y * img[dim_x * id_y_next + id_x] +
                                gap_x * (1 - gap_y) * img[dim_x * (id_y) + id_x_next] +
                                (1 - gap_x) * (1 - gap_y) * img[dim_x * id_y + id_x];
            }
            else{
                if ( gap_x < 0.5 ) id_x = index_x;
                else id_x = id_x_next;
                if ( gap_y < 0.5 ) id_y = index_y;
                else id_y = id_y_next;
                output[index] = img[dim_x * id_y + id_x];
            }
        }
    }
    __syncthreads();
}

__global__ void interplate_3D(float* coords, 
                                    float* img, 
                                    float* output,
                                    int order,
                                    size_t dim_z,
                                    size_t dim_y,
                                    size_t dim_x,
                                    int mode, float cval){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim_x * dim_y * dim_z;
    size_t dim_xy = dim_x * dim_y;

    if(index < total){
        float coords_z = coords[index];
        float coords_y = coords[index + total];
        float coords_x = coords[index + 2 * total];
        if(mode != 0){
            coords_z = lookup_second(lookup_first(coords_z, (int)dim_z, mode), dim_z);
            coords_y = lookup_second(lookup_first(coords_y, (int)dim_y, mode), dim_y);
            coords_x = lookup_second(lookup_first(coords_x, (int)dim_x, mode), dim_x);
        }
               
        int index_z = (int)floorf(coords_z);
        int index_y = (int)floorf(coords_y);
        int index_x = (int)floorf(coords_x);

        float gap_z = coords_z - index_z;
        float gap_y = coords_y - index_y;
        float gap_x = coords_x - index_x;

        int id_x = index_x, id_x_next = index_x + 1;
        int id_y = index_y, id_y_next = index_y + 1;
        int id_z = index_z, id_z_next = index_z + 1;
        
        if(mode == 0){
            if (index_y < 0 || index_y >= dim_y - 1 ||
                index_x < 0 || index_x >= dim_x - 1 || 
                index_z < 0 || index_z >= dim_z - 1 ){
                output[index] = cval;
            }
            else{
                if(order > 0)
                    output[index] =  gap_x * gap_y * gap_z * img[dim_xy * (index_z + 1) + dim_x * (index_y + 1) + index_x + 1] +
                                    (1 - gap_x) * gap_y * gap_z * img[dim_xy * (index_z + 1) + dim_x * (index_y + 1) + index_x] +
                                    gap_x * (1 - gap_y) * gap_z * img[dim_xy * (index_z + 1) + dim_x * (index_y) + index_x + 1] +
                                    (1 - gap_x) * (1 - gap_y) * gap_z * img[dim_xy * (index_z + 1) + dim_x * index_y + index_x] +
                                    gap_x * gap_y * (1 - gap_z) * img[dim_xy * index_z + dim_x * (index_y + 1) + index_x + 1] +
                                    (1 - gap_x) * gap_y * (1 - gap_z) * img[dim_xy * index_z + dim_x * (index_y + 1) + index_x] +
                                    gap_x * (1 - gap_y) * (1 - gap_z) * img[dim_xy * index_z + dim_x * (index_y) + index_x + 1] +
                                    (1 - gap_x) * (1 - gap_y) * (1 - gap_z) * img[dim_xy * index_z + dim_x * index_y + index_x];
                else{
                    if ( gap_x < 0.5 ) id_x = index_x;
                    else id_x = id_x_next;
                    if ( gap_y < 0.5 ) id_y = index_y;
                    else id_y = id_y_next;
                    if ( gap_z < 0.5 ) id_z = index_z;
                    else id_z = id_z_next;
                    output[index] = img[dim_xy * id_z + dim_x * id_y + id_x];
                }
            }
        }
        else{
            if(order > 0)
                output[index] =  gap_x * gap_y * gap_z * img[dim_xy * id_z_next + dim_x * id_y_next + id_x_next] +
                                (1 - gap_x) * gap_y * gap_z * img[dim_xy * id_z_next + dim_x * id_y_next + id_x] +
                                gap_x * (1 - gap_y) * gap_z * img[dim_xy * id_z_next + dim_x * (id_y) + id_x_next] +
                                (1 - gap_x) * (1 - gap_y) * gap_z * img[dim_xy * id_z_next + dim_x * id_y + id_x] +
                                gap_x * gap_y * (1 - gap_z) * img[dim_xy * id_z + dim_x * id_y_next + id_x_next] +
                                (1 - gap_x) * gap_y * (1 - gap_z) * img[dim_xy * id_z + dim_x * id_y_next + id_x] +
                                gap_x * (1 - gap_y) * (1 - gap_z) * img[dim_xy * id_z + dim_x * (id_y) + id_x_next] +
                                (1 - gap_x) * (1 - gap_y) * (1 - gap_z) * img[dim_xy * id_z + dim_x * id_y + id_x];
            else{
                if ( gap_x < 0.5 ) id_x = index_x;
                else id_x = id_x_next;
                if ( gap_y < 0.5 ) id_y = index_y;
                else id_y = id_y_next;
                if ( gap_z < 0.5 ) id_z = index_z;
                else id_z = id_z_next;
                output[index] = img[dim_xy * id_z + dim_x * id_y + id_x];
            }            
        }
    }
    __syncthreads();
}
