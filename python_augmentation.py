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
        import ipdb; ipdb.set_trace()
        return False

def test_3D():
    data_pth = 'data/FLAIR.nii.gz'
    sitk_image = sitk.ReadImage(data_pth)
    array_image = sitk.GetArrayFromImage(sitk_image).copy()

    cuda_handle = Handle(array_image.shape)
    # cuda_handle.scale(0.5)
    # cuda_handle.flip(do_y=True, do_x=True, do_z=True)
    cuda_handle.translate(100, 100, 20)
    cuda_handle.end_flag()

    correct_ret = deform.spatial_augment(array_image)
    # Warm up and Unit test
    for i in range(100):
        output = cuda_handle.augment(array_image)
    check(correct_ret, output[0])

    start = time.time()
    for i in range(Iters):
        output = cuda_handle.augment(array_image)
    end = time.time()
    print("Shape:{} Augmentation On CUDA Cost {}ms".format(array_image.shape, \
                                    (end - start) * 1000 / Iters))

    start = time.time()
    for i in range(Iters_CPU):
        correct_ret = deform.spatial_augment(array_image)
    end = time.time()
    print("Shape:{} Augmentation On CPU Cost {}ms".format(array_image.shape, \
                                    (end - start) * 1000 / Iters_CPU))

def test_2D():

    data_pth = 'data/Danny.jpg'
    image = Image.open(data_pth)
    array_image = np.array(image)
    raw = array_image
    array_image = array_image.transpose(2,0,1).astype(np.float32).copy()

    cuda_handle = Handle(array_image.shape, RGB=True)
    # cuda_handle.scale(0.5)
    # cuda_handle.flip(do_y=True)
    cuda_handle.translate(400, 400)
    cuda_handle.end_flag()

    if len(array_image.shape) == 2:
        correct_ret = deform.spatial_augment(array_image)
    else:
        correct_ret = np.zeros_like(array_image)
        for i in range(3):
            correct_ret[i] = deform.spatial_augment(array_image[i])

    # Warm up and Unit test
    for i in range(100):
        output = cuda_handle.augment(array_image)
    # check(correct_ret, output[0])
   
    # Save Image
    name, image_type = data_pth.split('.')
    for item in output[1]:
        name += '_' + item
    output_pth = name + '.' + image_type
    out = Image.fromarray(output[0].transpose(1, 2, 0), mode=image.mode)
    out = Image.fromarray(output[0].transpose(1, 2, 0).\
                                    astype(raw.dtype), mode=image.mode)
    out.save(output_pth)
    import ipdb; ipdb.set_trace()

    # Test Time
    start = time.time()
    for i in range(Iters):
        output = cuda_handle.augment(array_image)
    end = time.time()
    print("Shape:{} Augmentation On CUDA Cost {}ms".format(array_image.shape, \
                                    (end - start) * 1000 / Iters))

    start = time.time()
    for i in range(Iters_CPU):
        if len(array_image.shape) == 2:
            correct_ret = deform.spatial_augment(array_image)
        else:
            correct_ret = np.zeros_like(array_image)
            for i in range(3):
                correct_ret[i] = deform.spatial_augment(array_image[i])
    end = time.time()
    print("Shape:{} Augmentation On CPU Cost {}ms".format(array_image.shape, \
                                    (end - start) * 1000 / Iters_CPU)) 

if __name__ == "__main__":
    # test_3D()
    test_2D()
