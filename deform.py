
import numpy as np
import SimpleITK as sitk

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import map_coordinates

import time

def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float)) / 2.)[d]
    return coords

def elastic_deform_coordinates(coordinates, alpha, sigma):
    n_dim = len(coordinates)
    offsets = []
    for _ in range(n_dim):
        offsets.append(
            gaussian_filter((np.random.random(coordinates.shape[1:]) * 2 - 1), sigma, mode="constant", cval=0) * alpha)
    offsets = np.array(offsets)
    indices = offsets + coordinates
    return indices

if __name__ == "__main__":
    Iters = 10

    data_pth = 'data/FLAIR.nii.gz'
    sitk_image = sitk.ReadImage(data_pth)
    array_image = sitk.GetArrayFromImage(sitk_image).copy()

    elastic_time = 0.
    map_coordinates_time = 0.
    for i in range(Iters):
        start = time.time()
        coords = create_zero_centered_coordinate_mesh(array_image.shape)

        # alpha=(0., 1000.)
        # sigma=(10., 13.)
        # a = np.random.uniform(alpha[0], alpha[1])
        # s = np.random.uniform(sigma[0], sigma[1])
        # coords = elastic_deform_coordinates(coords, a, s)


        for d in range(len(array_image.shape)):
            ctr = int(np.round(array_image.shape[d] / 2.))
            coords[d] += ctr
        e_time = time.time()
        elastic_time += e_time - start

        ret = np.zeros_like(array_image)
        map_coordinates(ret.astype(float), coords, order=3, mode='mirror').astype(ret.dtype)
        m_time = time.time()
        map_coordinates_time += m_time - e_time

    print("Shape: {} \n elastic_deform: {}ms \n map_coordinates: {}ms"\
               .format(array_image.shape, elastic_time*1000/Iters, map_coordinates_time*1000/Iters))



    # cuda_handle = cuda_api.init_3D(array_image.shape[0], array_image.shape[1], array_image.shape[2])

    # for i in range(100):
    #     output = np.ones(array_image.shape).astype(np.float32)
    #     cuda_api.do_nothing(cuda_handle, output, array_image)
    # test = output == array_image
    # print(test)

    # start = time.time()
    # for i in range(Iters):
    #     output = np.ones(array_image.shape).astype(np.float32)
    #     # cuda_api.do_nothing(cuda_handle, output, array_image)
    #     output += array_image
    # end = time.time()
    # print("Shape:{} Cost {}ms".format(array_image.shape, \
    #                                 (end - start) * 1000 / Iters))
