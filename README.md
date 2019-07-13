# Cuda_Spatial_Deform

A fast tool to do image augmentation by CUDA on GPU(especially elastic deform), can be helpful to research on Medical Image.

## Motivation
- When the size of image is too large, it will take a lot of time(much more than forward and backward computation like U_Net), especially for 3D image(like CT).
- Elastic deformation on CPU is too slow.
- Doing Sptial_Deform by muti-processing consumes of too much CPU resources, whitch most GPU servers(like 32 cores with 4 gpus) can not afford.

## Implementation Overview
- Doing Spation_Deform on GPU instead of CPU, greatly saving CPU resources.
- Very Fast, speed up 25x at rotation, 45x at elastic_deform.
- Support many types of spatial deform: flip, rotate, scale, translate, elastic_deform.
- Support many rules of map_coordinates: mirror, constant, reflect, wrap, nearest.
- Doing Spation_Deform by doing calculates on coordinates, all transformations get combined before they are applied to the image
- Implement map_coordinates by linear interpolation.
- Unit test pass when over 99% pixels has L1_loss < 1e-3.
- Flexible with coordinates, users can fetch them form CUDA and do cubic interpolation at CPU by scipy.map_coordinates(order = 3)

## Speed Test
Test on 3D image , shape = [48, 240, 240]

Time(ms) | Rotate | Elastic
---  | --- | ---
CUDA | 14 | 40
CPU | 304 | 1821

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
    mode: The rules of map_coordinates. Refernce to  https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
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
```python
# The shape must be equal to cuda_handle.shape
array_image = load_np_array(data_pth)
output = cuda_handle.augment(array_image)
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
