#include "mode_index.cuh"

__device__ float lookup_second(float index, int max_dim){
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

__device__ float lookup_first(float index, int max_dim, int mode){
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
