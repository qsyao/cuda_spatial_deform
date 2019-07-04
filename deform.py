
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

def create_matrix_rotation_x_3d(angle, matrix=None):
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
    if matrix is None:
        return rotation_x

    return np.dot(matrix, rotation_x)


def create_matrix_rotation_y_3d(angle, matrix=None):
    rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                           [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]])
    if matrix is None:
        return rotation_y

    return np.dot(matrix, rotation_y)


def create_matrix_rotation_z_3d(angle, matrix=None):
    rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]])
    if matrix is None:
        return rotation_z

    return np.dot(matrix, rotation_z)

def rotate_coords_3d(coords, angle_x, angle_y, angle_z):
    rot_matrix = np.identity(len(coords))
    rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
    rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
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

        alpha=(0., 1000.)
        sigma=(10., 13.)
        a = np.random.uniform(alpha[0], alpha[1])
        s = np.random.uniform(sigma[0], sigma[1])
        coords = elastic_deform_coordinates(coords, a, s)

        # angle_x=(0, 2 * np.pi)
        # angle_y=(0, 2 * np.pi)
        # angle_z=(0, 2 * np.pi)
        # if angle_x[0] == angle_x[1]:
        #     a_x = angle_x[0]
        # else:
        #     a_x = np.random.uniform(angle_x[0], angle_x[1])
        # if angle_y[0] == angle_y[1]:
        #     a_y = angle_y[0]
        # else:
        #     a_y = np.random.uniform(angle_y[0], angle_y[1])
        # if angle_z[0] == angle_z[1]:
        #     a_z = angle_z[0]
        # else:
        #     a_z = np.random.uniform(angle_z[0], angle_z[1])
        # coords = rotate_coords_3d(coords, a_x, a_y, a_z)

        # for d in range(len(array_image.shape)):
        #     ctr = int(np.round(array_image.shape[d] / 2.))
        #     coords[d] += ctr
        e_time = time.time()
        elastic_time += e_time - start

        ret = np.zeros_like(array_image)
        map_coordinates(ret.astype(float), coords, order=1, mode='mirror').astype(ret.dtype)
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
