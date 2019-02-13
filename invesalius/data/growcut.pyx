import numpy as np
cimport numpy as np
cimport cython

from collections import deque

from cython.parallel import prange
from libc.math cimport floor, ceil, fabs
from libc.string cimport memcpy
#  from libcpp cimport bool
#  from libcpp.deque cimport deque as cdeque
#  from libcpp.vector cimport vector

from cy_my_types cimport image_t, mask_t

cdef struct s_coord:
    int x
    int y
    int z

ctypedef s_coord coord


@cython.nonecheck(False)
@cython.cdivision(True) 
cdef inline float g(float x, image_t _min, image_t _max) nogil:
    cdef float value = x - _min
    return 1.0 - value / <float>(_max - _min)


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
@cython.nonecheck(False)
def growcut_cellular_automata(np.ndarray[image_t, ndim=3] data, mask_t[:, :, :] mask, mask_t[:, :, :] strct, mask_t[:, :, :] out):

    cdef int to_return = 0
    if out is None:
        out = np.zeros_like(data)
        to_return = 1

    cdef int x, y, z
    cdef int dx, dy, dz
    cdef int odx, ody, odz
    cdef int xo, yo, zo
    cdef int i, j, k
    cdef int offset_x, offset_y, offset_z
    cdef int modified = 1
    cdef int vmodified = 0
    cdef int qte = 0

    #  print("data shape", data.shape)
    cdef float[:, :, :] weights0 = np.zeros_like(data, dtype='float32')
    cdef float[:, :, :] weights1 = np.zeros_like(data, dtype='float32')
    cdef float weight = 0

    cdef float **mweights = [&weights0[0, 0, 0], &weights1[0, 0, 0]]

    cdef int aweight = 0
    cdef int nweight = 1

    cdef image_t _min = data.min()
    cdef image_t _max = data.max()

    print("Min max", _min, _max)

    memcpy(&out[0, 0, 0], &mask[0, 0, 0], mask.size)

    dz = data.shape[0]
    dy = data.shape[1]
    dx = data.shape[2]

    odz = strct.shape[0]
    ody = strct.shape[1]
    odx = strct.shape[2]

    offset_z = odz / 2
    offset_y = ody / 2
    offset_x = odx / 2

    for z in prange(dz, nogil=True):
        for y in range(dy):
            for x in xrange(dx):
                if mask[z, y, x]:
                    weights0[z, y , x] = 1
                    weights1[z, y , x] = 1


    while modified:
        modified = 0
        for z in prange(dz, nogil=True):
            for y in range(dy):
                for x in xrange(dx):
                    vmodified = 0
                    for k in xrange(odz):
                        zo = z + k - offset_z
                        for j in xrange(ody):
                            yo = y + j - offset_y
                            for i in xrange(odx):
                                if strct[k, j, i]:
                                    xo = x + i - offset_x
                                    if 0 <= xo < dx and 0 <= yo < dy and 0 <= zo < dz:
                                        weight = g(fabs(data[z, y, x] - data[zo, yo, xo]), _min, _max) * mweights[aweight][zo*dx*dy + yo*dx + xo]
                                        if weight > mweights[aweight][z*dx*dy + y*dx + x]:
                                            out[z, y, x] = out[zo, yo, xo]
                                            mweights[nweight][z*dx*dy + y*dx + x] = weight
                                            #  with gil:
                                                #  print(aweight, nweight, "weights", weight, weights0[z, y, x], weights1[z, y, x])
                                            modified += 1
                                            vmodified = 1
                    if not vmodified:
                        mweights[nweight][z*dx*dy + y*dx + x] = mweights[aweight][z*dx*dy + y*dx + x]
        #  memcpy(&weights0[0, 0, 0], &weights1[0, 0, 0], dx * dy * dz * 4)
        aweight = nweight
        nweight = (nweight + 1) % 2
        qte += 1

    print("Rodou", qte, "vezes")

    if to_return:
        return out
