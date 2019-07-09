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
l_i.argtypes = [c_void_p, ndpointer(np.float32), ndpointer(np.float32), c_int]

check = lib.check_coords
check.argtypes = [c_void_p, ndpointer(np.float32)]

scale = lib.cu_scale
scale.argtypes = [c_void_p, c_float]

end_flag = lib.endding_flag
end_flag.argtypes = [c_void_p]

flip = lib.cu_flip
flip.argtypes = [c_void_p, c_int, c_int, c_int]

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

def bool_to_int(flag):
    if flag:
        return 1
    else:
        return 0

class Flip(Spatial_Deform):
    def __init__(self, do_x=False, do_y=False, do_z=False, prob=1.0):
        Spatial_Deform.__init__(self, prob)
        self.label = 'Flip'
        self.do_x = bool_to_int(do_x)
        self.do_y = bool_to_int(do_y)
        self.do_z = bool_to_int(do_z)
    
    def defrom(self, handle):
        if np.random.uniform() < self.prob:
            flip(handle, self.do_x, self.do_y, self.do_z)
            return self.label
        else:
            return None

class End_Flag(Spatial_Deform):
    def __init__(self, prob=1.0):
        Spatial_Deform.__init__(self, prob)

    def defrom(self, handle):
        end_flag(handle)
        return None

class Handle(object):
    def __init__(self, shape, RGB=False):
        self.RGB = RGB
        self.shape = shape
        if self.RGB:
            self.shape = shape[1:]
            
        self.deform_list = []

        if(len(shape) != 2 and len(shape) != 3):
            raise ValueError
        if(len(shape) == 2 or RGB):
            self.cuda_handle = init_2D(self.shape[0], self.shape[1])
            self.is_3D = False
        else:
            self.is_3D = True
            self.cuda_handle = init_3D(shape[0], shape[1],  shape[2])
    
    def augment(self, img):
        if self.RGB:
            assert(img.shape[0] == 3)
            assert(img.shape[1:] == self.shape)
        else:
            assert(img.shape == self.shape)
        output = np.ones(img.shape).astype(np.float32)
        labels = self.deform_coords()
        
        # check coords
        self.get_coords()

        if not self.RGB:
            l_i(self.cuda_handle, output, img, 1)
        else:
            for i in range(3):
                if i == 2:
                    l_i(self.cuda_handle, output[i], img[i], 1)
                else:
                    l_i(self.cuda_handle, output[i], img[i], 0)

        return [output, labels]

    def scale(self, sc, prob=1.0):
        self.deform_list.append(Scale(sc, prob))
    
    def flip(self, do_x=False, do_y=False, do_z=False, prob=1.0):
        self.deform_list.append(Flip(do_x, do_y, do_z, prob))

    def end_flag(self):
        self.deform_list.append(End_Flag())

    def get_coords(self):
        coords_shape = list(self.shape)
        coords_shape.insert(0, 3 if self.is_3D else 2)
        coords = np.ones(coords_shape).astype(np.float32)
        check(self.cuda_handle, coords)
        coords = coords.astype(np.int)
        import ipdb; ipdb.set_trace()
        return coords

    def deform_coords(self):
        labels = []
        for item in self.deform_list:
            out_label = item.defrom(self.cuda_handle)
            if out_label is not None:
                labels.append(out_label)
        return labels
