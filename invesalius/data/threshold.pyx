import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport floor, ceil, sqrt, fabs
from cython.parallel import prange

DTYPE8 = np.uint8
ctypedef np.uint8_t DTYPE8_t

DTYPE16 = np.int16
ctypedef np.int16_t DTYPE16_t

DTYPEF32 = np.float32
ctypedef np.float32_t DTYPEF32_t

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
@cython.wraparound(False)
def threshold(DTYPE16_t[:, :] image, DTYPE8_t[:, :] mask, DTYPE16_t low, DTYPE16_t high):
    cdef int sy = image.shape[0]
    cdef int sx = image.shape[1]
    cdef int x, y
    cdef DTYPE16_t v
    for y in prange(sy, nogil=True):
        for x in xrange(sx):
            v = image[y, x]
            if not mask[y, x]:
                if v >= low and v <= high:
                    mask[y, x] = 255
                else:
                    mask[y, x] = 0
