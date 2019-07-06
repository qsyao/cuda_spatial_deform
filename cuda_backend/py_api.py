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

l_i = lib.linear_interpolate
l_i.argtypes = [c_void_p, ndpointer(np.float32), ndpointer(np.float32)]

check = lib.check_coords
check.argtypes = [c_void_p, ndpointer(np.float32)]

scale = lib.cu_scale
scale.argtypes = [c_void_p, c_float]

class Spatial_Deform(object):
    def __init__(self, prob=1.0):
        self.prob = prob
    
    def defrom(self):
        pass

class Scale(Spatial_Deform):
    def __init__(self, sc, prob=1.0):
        Spatial_Deform.__init__(self, prob)
        assert(sc > 0 and sc <= 1.0)
        self.sc = sc
        self.label = 'Scale'
    
    def defrom(self, handle):
        if np.random.uniform() < self.prob:
            scale(handle, self.sc)
            return self.label
        else:
            return None

class Handle(object):
    def __init__(self, shape):
        self.shape = shape
        self.deform_list = []
        if(len(shape) != 2 and len(shape) != 3):
            raise ValueError
        if(len(shape) == 2):
            self.cuda_handle = init_2D(shape[0],  shape[1])
            self.is_3D = False
        else:
            self.is_3D = True
            self.cuda_handle = init_3D(shape[0], shape[1],  shape[2])
    
    def interpolate(self, img):
        assert(img.shape == self.shape)
        output = np.ones(self.shape).astype(np.float32)
        labels = self.deform_coords()
        l_i(self.cuda_handle, output, img)
        return [output, labels]

    def scale(self, sc, prob=1.0):
        self.deform_list.append(Scale(sc, prob))

    def get_coords(self):
        coords_shape = list(self.shape)
        coords_shape.insert(0, 3 if self.is_3D else 2)
        coords = np.ones(coords_shape).astype(np.float32)
        check(self.cuda_handle, coords)
        return coords

    def deform_coords(self):
        labels = []
        for item in self.deform_list:
            out_label = item.defrom(self.cuda_handle)
            if out_label is not None:
                labels.append(out_label)
        return labels
