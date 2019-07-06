#include "interpolate.cuh"
#include <math.h>

__global__ void linear_interplate_2D(float* coords, 
                                    float* img, 
                                    float* output, 
                                    size_t dim_y,
                                    size_t dim_x){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < dim_x * dim_y){
        // recenter
        coords[index] += (float)dim_y/2.0;
        coords[index + dim_x*dim_y] += (float)dim_x/2.0;

        float coords_y = coords[index];
        float coords_x = coords[index + dim_x*dim_y];
        int index_y = (int)floorf(coords_y);
        int index_x = (int)floorf(coords_x);
        float gap_y = coords_y - index_y;
        float gap_x = coords_x - index_x;

        if (index_y < 0 || index_y >= dim_y -1 ||
            index_x < 0 || index_x >= dim_x -1 ){
            output[index] = 0;
        }
        else{
            output[index] = gap_x * gap_y * img[dim_x * (index_y + 1) + index_x + 1] +
                            (1 - gap_x) * gap_y * img[dim_x * (index_y + 1) + index_x] +
                            gap_x * (1 - gap_y) * img[dim_x * (index_y) + index_x + 1] +
                            (1 - gap_x) * (1 - gap_y) * img[dim_x * index_y + index_x];
        }
    }
    __syncthreads();
}

__global__ void linear_interplate_3D(float* coords, 
                                    float* img, 
                                    float* output,
                                    size_t dim_z,
                                    size_t dim_y,
                                    size_t dim_x){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim_x * dim_y * dim_z;
    size_t dim_xy = dim_x * dim_y;

    if(index < total){
        // recenter
        coords[index] += (float)dim_z/2.0;
        coords[index + total] += (float)dim_y/2.0;
        coords[index + 2 * total] += (float)dim_x/2.0;
   
        float coords_z = coords[index];
        float coords_y = coords[index + total];
        float coords_x = coords[index + 2 * total];
        int index_z = (int)floorf(coords_z);
        int index_y = (int)floorf(coords_y);
        int index_x = (int)floorf(coords_x);
        float gap_z = coords_z - index_z;
        float gap_y = coords_y - index_y;
        float gap_x = coords_x - index_x;

        if (index_y < 0 || index_y >= dim_y - 1 ||
            index_x < 0 || index_x >= dim_x - 1 || 
            index_z < 0 || index_z >= dim_z - 1 ){
            output[index] = 0.0;
        }
        else{
            output[index] =  gap_x * gap_y * gap_z * img[dim_xy * (index_z + 1) + dim_x * (index_y + 1) + index_x + 1] +
                            (1 - gap_x) * gap_y * gap_z * img[dim_xy * (index_z + 1) + dim_x * (index_y + 1) + index_x] +
                            gap_x * (1 - gap_y) * gap_z * img[dim_xy * (index_z + 1) + dim_x * (index_y) + index_x + 1] +
                            (1 - gap_x) * (1 - gap_y) * gap_z * img[dim_xy * (index_z + 1) + dim_x * index_y + index_x] +
                            gap_x * gap_y * (1 - gap_z) * img[dim_xy * index_z + dim_x * (index_y + 1) + index_x + 1] +
                            (1 - gap_x) * gap_y * (1 - gap_z) * img[dim_xy * index_z + dim_x * (index_y + 1) + index_x] +
                            gap_x * (1 - gap_y) * (1 - gap_z) * img[dim_xy * index_z + dim_x * (index_y) + index_x + 1] +
                            (1 - gap_x) * (1 - gap_y) * (1 - gap_z) * img[dim_xy * index_z + dim_x * index_y + index_x];
        }
    }
    __syncthreads();
}
