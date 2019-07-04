from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer 

import os
lib_dir = os.path.abspath(os.path.dirname(__file__))
lib = CDLL(lib_dir + "/libcudaAugmentation.so", RTLD_GLOBAL)

init_2D = lib.init_2D_handle
init_2D.argtypes = [c_int, c_int]
init_2D.restype = c_void_p

init_3D = lib.init_3D_handle
init_3D.argtypes = [c_int, c_int, c_int]
init_3D.restype = c_void_p

cu_do_nothing = lib.test
cu_do_nothing.argtypes = [c_void_p, ndpointer(np.float32), \
                            ndpointer(np.float32)]

check = lib.check_coords
check.argtypes = [c_void_p, ndpointer(np.float32)]

class Handle(object):
    def __init__(self, shape):
        self.shape = shape
        if(len(shape) != 2 and len(shape) != 3):
            raise ValueError
        if(len(shape) == 2):
            self.cuda_handle = init_2D(shape[0],  shape[1])
            self.is_3D = False
        else:
            self.is_3D = True
            self.cuda_handle = init_3D(shape[0], shape[1],  shape[2])
    
    def do_nothing(self, img):
        assert(img.shape == self.shape)
        output = np.ones(self.shape).astype(np.float32)
        cu_do_nothing(self.cuda_handle, output, img)
        return output
    
    def check_coords(self):
        coords_shape = list(self.shape)
        coords_shape.insert(0, 3 if self.is_3D else 2)
        coords = np.ones(coords_shape).astype(np.float32)
        check(self.cuda_handle, coords)
        return coords
