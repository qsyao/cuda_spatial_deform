from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer 

import os
lib_dir = os.path.abspath(os.path.dirname(__file__))
lib = CDLL(lib_dir + "/libcudaAugmentation.so", RTLD_GLOBAL)

init_2D = lib.init_2D_handle
init_2D.argtypes = [c_int, c_int, c_int, c_float]
init_2D.restype = c_void_p

init_3D = lib.init_3D_handle
init_3D.argtypes = [c_int, c_int, c_int, c_int, c_float]
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

translate = lib.cu_translate
translate.argtypes = [c_void_p, c_float, c_float, c_float]

rotate_2D = lib.cu_rotate_2D
rotate_2D.argtypes = [c_void_p, c_float]

rotate_3D = lib.cu_rotate_3D
rotate_3D.argtypes = [c_void_p, ndpointer(np.float32)]

elastic = lib.cu_elastic
elastic.argtypes = [c_void_p, c_float, c_float, c_float, c_int, c_float]

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

class Translate(Spatial_Deform):
    def __init__(self, seg_x=0.0, seg_y=0.0, seg_z=0.0, prob=1.0):
        Spatial_Deform.__init__(self, prob)
        self.label = 'Translate'
        self.seg_x = seg_x
        self.seg_y = seg_y
        self.seg_z = seg_z
    
    def defrom(self, handle):
        if np.random.uniform() < self.prob:
            translate(handle, self.seg_x, self.seg_y, self.seg_z)
            return self.label
        else:
            return None

class Rotate(Spatial_Deform):
    def __init__(self, is_2D, angel_x, angle_y=0, angle_z=0, prob=1.0):
        Spatial_Deform.__init__(self, prob)
        self.label = 'Rotate'
        self.is_2D = is_2D
        self.angel_x = angel_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.matrix = np.identity(3).astype(np.float32)
        if not is_2D:
            rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angel_x), -np.sin(angel_x)],
                           [0, np.sin(angel_x), np.cos(angel_x)]]).astype(np.float32)
            rotation_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                           [0, 1, 0],
                           [-np.sin(angle_y), 0, np.cos(angle_y)]]).astype(np.float32)
            rotation_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                           [np.sin(angle_z), np.cos(angle_z), 0],
                           [0, 0, 1]]).astype(np.float32)
            self.matrix = np.dot(self.matrix, rotation_x)
            self.matrix = np.dot(self.matrix, rotation_y)
            self.matrix = np.dot(self.matrix, rotation_z)        
    
    def defrom(self, handle):
        if np.random.uniform() < self.prob:
            if self.is_2D:
                rotate_2D(handle, self.angel_x)
                return self.label
            else:
                rotate_3D(handle, self.matrix)
                return self.label
        else:
            return None

class Elastic(Spatial_Deform):
    def __init__(self, sigma, alpha, mode='reflect', \
                                    c_val=0, truncate=4.0, prob=1.0):
        Spatial_Deform.__init__(self, prob)
        self.label = 'Elastic'
        self.sigma = sigma
        self.alpha = alpha
        self.c_val = c_val
        self.truncate = truncate
        type_mode = -1
        if mode == 'constant':
            type_mode = 0
        elif mode == 'mirror':
            type_mode = 1
        else:
            raise ValueError
        self.type_mode = type_mode

    def defrom(self, handle):
        if np.random.uniform() < self.prob:
            elastic(handle, self.sigma, self.alpha, self.truncate, self.type_mode, self.c_val)
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
    def __init__(self, shape, RGB=False, mode='constant', cval=0.0):
        self.RGB = RGB
        self.shape = shape
        if self.RGB:
            self.shape = shape[1:]
            
        self.deform_list = []

        if(len(shape) != 2 and len(shape) != 3):
            raise ValueError
        
        type_mode = -1
        if mode == 'constant':
            type_mode = 0
        elif mode == 'reflect':
            type_mode = 1
        elif mode == 'mirror':
            type_mode = 2
        elif mode == 'nearest':
            type_mode = 3
        elif mode == 'wrap':
            type_mode = 4
        else:
            raise ValueError

        if(len(shape) == 2 or RGB):
            self.cuda_handle = init_2D(self.shape[0], self.shape[1], type_mode, float(cval))
            self.is_3D = False
        else:
            self.is_3D = True
            self.cuda_handle = init_3D(shape[0], shape[1],  shape[2], type_mode, float(cval))
    
    def augment(self, img):
        if self.RGB:
            assert(img.shape[0] == 3)
            assert(img.shape[1:] == self.shape)
        else:
            assert(img.shape == self.shape)
        output = np.ones(img.shape).astype(np.float32)
        labels = self.deform_coords()
        
        # # check coords
        # self.get_coords()

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

    def translate(self, seg_x=0.0, seg_y=0.0, seg_z=0.0, prob=1.0):
        self.deform_list.append(Translate(seg_x, seg_y, seg_z, prob))
    
    def rotate(self, angel_x=0, angle_y=0, angle_z=0, prob=1.0):
        self.deform_list.append(Rotate(not self.is_3D, angel_x, angle_y, angle_z, prob))

    def elastic(self, sigma, alpha, mode='reflect', c_val=0, truncate=4.0, prob=1.0):
        self.deform_list.append(Elastic(sigma, alpha, mode, c_val, truncate, prob))

    def end_flag(self):
        self.deform_list.append(End_Flag())

    def get_coords(self):
        coords_shape = list(self.shape)
        coords_shape.insert(0, 3 if self.is_3D else 2)
        coords = np.ones(coords_shape).astype(np.float32)
        check(self.cuda_handle, coords)
        # coords = coords.astype(np.int)
        import ipdb; ipdb.set_trace()
        return coords

    def deform_coords(self):
        labels = []
        for item in self.deform_list:
            out_label = item.defrom(self.cuda_handle)
            if out_label is not None:
                labels.append(out_label)
        return labels
