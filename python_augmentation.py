import SimpleITK as sitk
import numpy as np
import time

from cuda_backend.py_api import Handle

Iters = 100

def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float)) / 2.)[d]
    return coords

if __name__ == "__main__":
    data_pth = 'data/FLAIR.nii.gz'
    sitk_image = sitk.ReadImage(data_pth)
    array_image = sitk.GetArrayFromImage(sitk_image).copy()
    cuda_handle = Handle(array_image.shape)

    for i in range(100):
        output = cuda_handle.do_nothing(array_image)
    test = output == array_image
    print(test)


    coords = cuda_handle.check_coords()
    cor_coords = create_zero_centered_coordinate_mesh(array_image.shape)
    import ipdb; ipdb.set_trace()

    start = time.time()
    for i in range(Iters):
        output = np.ones(array_image.shape).astype(np.float32)
        # output = cuda_handle.do_nothing(array_image)
        output += array_image
    end = time.time()
    print("Shape:{} Cost {}ms".format(array_image.shape, \
                                    (end - start) * 1000 / Iters))
