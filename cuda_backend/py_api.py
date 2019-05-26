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

do_nothing = lib.test
do_nothing.argtypes = [c_void_p, ndpointer(np.float32), \
                            ndpointer(np.float32)]
