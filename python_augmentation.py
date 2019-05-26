import SimpleITK as sitk
import numpy as np
import time

import cuda_backend.py_api as cuda_api

Iters = 100

if __name__ == "__main__":
    data_pth = 'data/FLAIR.nii.gz'
    sitk_image = sitk.ReadImage(data_pth)
    array_image = sitk.GetArrayFromImage(sitk_image).copy()
    cuda_handle = cuda_api.init_3D(array_image.shape[0], array_image.shape[1], array_image.shape[2])

    for i in range(100):
        output = np.ones(array_image.shape).astype(np.float32)
        cuda_api.do_nothing(cuda_handle, output, array_image)
    test = output == array_image
    print(test)

    start = time.time()
    for i in range(Iters):
        output = np.ones(array_image.shape).astype(np.float32)
        # cuda_api.do_nothing(cuda_handle, output, array_image)
        output += array_image
    end = time.time()
    print("Shape:{} Cost {}ms".format(array_image.shape, \
                                    (end - start) * 1000 / Iters))
