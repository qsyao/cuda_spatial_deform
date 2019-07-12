# Cuda_Spatial_Deform

## How to Use

### CMake
```shell
cd cuda_backend
cmake -D /path/to/cuda .
make -j8
```

### Set_Config
```python
# Import cuda_spation_deform Handle
from cuda_spatial_defrom import Cuda_Spatial_Deform

# Init Handle
cuda_handle = Handle(array_image.shape, mode="constant")
'''
    Shape: cuda_backend will malloc according to shape
    RGB: bool (Only Support 2D-RGB)
    mode: The rules of map_coordinates. Refernce to  https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
    cval: default is 0.0. Only be useful when mode == 'constant'
    id_gpu: choose the number of GPU
'''

# Choose your Rules of spatial_defrom

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

## Class Cuda_Spatial_Deform
```python
def __init__(self, shape, RGB=False, mode='constant', cval=0.0, id_gpu=0):
```
- shape: Support: 3D 2D RGB-2D
- mode: Support: 'constant' 'reflect' 'mirror' 'wrap' 'nearest'

```python
def scale(self, sc, prob=1.0):
```
- sc : rate of Scale , must in [0, 1]
- prob : The probability of doing this augmentation

```python
def translate(self, seg_x=0.0, seg_y=0.0, seg_z=0.0, prob=1.0):
```
- seg_x : coordinates += seg_x when dim == x
- prob : The probability of doing this augmentation

```python
def flip(self, do_x=False, do_y=False, do_z=False, prob=1.0):
```
- do_x. etc : bool default is False
- prob : The probability of doing this augmentation

```python
def rotate(self, angel_x=0, angle_y=0, angle_z=0, prob=1.0):
```
- angel_x. etc : Clockwise rotation at dim_x with angle_x
- prob : The probability of doing this augmentation

```python
def elastic(self, sigma, alpha, mode='constant', c_val=0, truncate=4.0, prob=1.0):
```
- sigma, alpha : coordinates += guassian_fiter(np.random.random(shape) * 2 - 1, sigma, truncate) * alpha
- mode : Rules of padding , only support 'mirror' and 'constant'
- prob : The probability of doing this augmentation

```python
def end_flag(self):
```
Must be added at the end of translations.

```python
def get_coords(self):
```
Return the coodinates. Can be used as cubic interpolation with scipy.map_coordinates.

```python
def reset(self):
```
Call to redifine the translations.

```python
def deform_coords(self):
```
Only calculates on coordinates.

```python
def augment(self, img):
```
- assert(img.shape == self.shape)
- return the list of output_array and done_list 
