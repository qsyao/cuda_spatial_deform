import SimpleITK as sitk
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

from cuda_backend.py_api import Handle
import deform

Iters = 1000
Iters_CPU = 1

def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float)) / 2.)[d]
    return coords

def check(correct, output):
    assert(correct.shape == output.shape)
    max_loss = np.abs(output - correct).max()
    if max_loss < 1e-4:
        print("Unit_test Successful Pass: Max loss < 0.0001")
        return True
    else:
        print("Unit_test Failed: Max loss is {}".format(max_loss))
        return False

if __name__ == "__main__":
    data_pth = 'data/FLAIR.nii.gz'
    sitk_image = sitk.ReadImage(data_pth)
    array_image = sitk.GetArrayFromImage(sitk_image).copy()

    # data_pth = 'data/Danny.jpg'
    # image = Image.open(data_pth)
    # # plt.imshow(image)
    # # plt.show()
    # array_image = np.array(image)
    # import ipdb; ipdb.set_trace()
    cuda_handle = Handle(array_image.shape)

    correct_ret = deform.spatial_augment(array_image)
    # Warm up and Unit test
    for i in range(100):
        output = cuda_handle.test(array_image, 0.5)
    check(correct_ret, output)

    start = time.time()
    for i in range(Iters):
        output = cuda_handle.test(array_image, 0.5)
    end = time.time()
    print("Shape:{} Augmentation On CUDA Cost {}ms".format(array_image.shape, \
                                    (end - start) * 1000 / Iters))

    start = time.time()
    for i in range(Iters_CPU):
        correct_ret = deform.spatial_augment(array_image)
    end = time.time()
    print("Shape:{} Augmentation On CPU Cost {}ms".format(array_image.shape, \
                                    (end - start) * 1000 / Iters_CPU))
