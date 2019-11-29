# Cuda_Spatial_Deform

A fast tool to do image augmentation by CUDA on GPU(especially elastic deformation), can be helpful to research on Medical Image Analysis.

## Motivation
- When the size of image is too large, it takes a lot of time(much more than forward and backward computation say in U_Net), especially for 3D image(like CT).
- Elastic deformation on CPU is too slow.
- Doing Sptial_Deform by muti-processing consumes of too much CPU resources, to which most GPU servers(like 32 cores with 4 gpus) can not afford.

## Implementation Overview
- Doing Spation_Deform on GPU instead of CPU, greatly saving CPU resources.
- Very Fast, speed up 25x for rotation, 45x for elastic_deformation.
- Support many types of spatial deform: flip, rotate, scale, translate, elastic_deformation.
- Support many rules of map_coordinates: mirror, constant, reflect, wrap, nearest.
- Doing Spatial_Deform by doing calculations of coordinates, all transformations get combined before they are applied to the image
- Implement map_coordinates by linear interpolation(for image) and the nearest interpolation(for labels).
- Unit test passes when over 99% pixels has L1_loss < 1e-3.
- Users can fetch coordinates from CUDA and do cubic interpolation at CPU by scipy.map_coordinates(order = 3)

## Speed Test
Test on 3D image , shape = [48, 240, 240]

Time(ms) | Rotate | Elastic
---  | --- | ---
CUDA | 14 | 40
CPU | 304 | 1821

## Citation
If you use our code, please cite our paper:

Chao Huang, Hu Han, Qingsong Yao, Shankuan Zhu, S. Kevin Zhou. , 3D U<sup>2</sup>-Net: A 3D Universal U-Net for Multi-Domain Medical Image Segmentation, MICCAI 2019.


## How to Use

### CMake
```shell
cd cuda_backend
cmake -D CUDA_TOOLKIT_ROOT_DIR=/path/to/cuda .
make -j8
```

### Set_Config
```python
# Import cuda_spation_deform Handle
from cuda_spatial_deform import Cuda_Spatial_Deform

# Init Handle
cuda_handle = Cuda_Spatial_Deform(array_image.shape, mode="constant")
'''
    Shape: cuda_backend will malloc according to shape
    RGB: bool (Only Support 2D-RGB)
    mode: The rules of map_coordinates. Reference  to  https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
    cval: default is 0.0. Only be useful when mode == 'constant'
    id_gpu: choose the number of GPU
'''

# Choose your Rules of spatial_deform

# cuda_handle.scale(0.5)
# cuda_handle.flip(do_y=True, do_x=True, do_z=True)
# cuda_handle.translate(100, 100, 20)
# cuda_handle.rotate(0.75 * np.pi, 0.75 * np.pi, 0.75 * np.pi)
cuda_handle.elastic(sigma=12., alpha=200., mode='constant')
cuda_handle.end_flag()
```

### DO augmentation
Warning: The numpy array that python frontend deliverd to the cuda backend must be continuous and have the correct order of dimensions in the real memory.
So if you want to use function(transpose) to change the shape of array, you must use array.transpose().copy().
Info: There are two step to do augmentation. You can lookup the functions in doc.md.
1 Deform coordinates stored in cuda backend by computation flow definated above.
2 Interpolate the input array with coordinates
3(optional) Reset the coordinates.
```python
# The shape must be equal to cuda_handle.shape
array_image = load_np_array(data_pth)
output = cuda_handle.augment(array_image, order=1)
# done_list will list the translations actually done
done_list = output[1]
output_array = output[0]
```

## Example_Image

### Flip
![Flip](https://github.com/qsyao/cuda_spatial_deform/blob/master/data/Daenerys_Flip.jpg)
### Rotate
![Rotate](https://github.com/qsyao/cuda_spatial_deform/blob/master/data/Daenerys_Rotate.jpg)
### Translate
![Translate](https://github.com/qsyao/cuda_spatial_deform/blob/master/data/Daenerys_Translate.jpg)
### Scale
![Scale](https://github.com/qsyao/cuda_spatial_deform/blob/master/data/Daenerys_Scale.jpg)
### Elastic_Deform
![Elastic_Deform](https://github.com/qsyao/cuda_spatial_deform/blob/master/data/Daenerys_Elastic.jpg)
## Reference
[batchgenerators](https://github.com/MIC-DKFZ/batchgenerators)

[scipy](https://github.com/scipy/scipy)

The elastic deformation approach is described in
*   Ronneberger, Fischer, and Brox, "U-Net: Convolutional Networks for Biomedical
    Image Segmentation" (<https://arxiv.org/abs/1505.04597>)
*   Çiçek et al., "3D U-Net: Learning Dense Volumetric
    Segmentation from Sparse Annotation" (<https://arxiv.org/abs/1606.06650>)
