import numpy as np
cimport numpy as np
cimport cython

#  ctypedef np.uint16_t image_t
#
from libcpp.unordered_map cimport unordered_map
from libcpp.set cimport set
from libcpp.vector cimport vector
from libcpp cimport bool

ctypedef fused image_t:
    np.float64_t
    np.int16_t
    np.uint8_t

ctypedef np.uint8_t mask_t

ctypedef np.float32_t vertex_t
ctypedef np.float32_t normal_t

# To compile in windows 64
IF UNAME_MACHINE == 'AMD64':
    ctypedef np.int64_t vertex_id_t
ELSE:
    ctypedef np.int_t vertex_id_t

