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
@cython.cdivision(True) 
cdef int _growcut_cellular_automata(image_t[:] data, mask_t[:] mask, int dx, int dy, int dz, mask_t[:] strct, int odx, int ody, int odz, image_t _min, image_t _max, float **mweights, mask_t[:] out) nogil:
    cdef int i, j, x, y, z, xo, yo, zo, sx, sy, sz, sj
    cdef int vmodified = 0
    cdef int modified = 1
    cdef int qte = 0

    cdef int di = dx * dy * dz
    cdef int dj = odx *ody * odz

    cdef int offset_z = odz / 2
    cdef int offset_y = ody / 2
    cdef int offset_x = odx / 2

    cdef float weight = 0


    cdef int aweight = 0
    cdef int nweight = 1

    for i in prange(di):
        if mask[i]:
            mweights[0][i] = 1
            mweights[1][i] = 1

    while modified:
        modified = 0

        for i in prange(di):
            vmodified = 0
            z = i / (dx * dy)
            y = i / (dx)
            x = i % dx
            for j in range(dj):
                if strct[j]:
                    sz = j / (odx * ody)
                    sy = j / (odx)
                    sx = j % (odx)

                    zo = z + sz - offset_z
                    yo = y + sy - offset_y
                    xo = x + sx - offset_x

                    if 0 <= xo < dx and 0 <= yo < dy and 0 <= zo < dz:
                        sj = zo * dx * dy + yo * dx + xo
                        weight = g(fabs(data[i] - data[sj]), _min, _max) * mweights[aweight][sj]

                        if weight > mweights[aweight][i]:
                            out[i] = out[sj]
                            mweights[nweight][i] = weight
                            modified += 1
                            vmodified = 1
            if not vmodified:
                mweights[nweight][i] = mweights[aweight][i]

        aweight = nweight
        nweight = (nweight + 1) % 2
        qte += 1

    return qte


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def growcut_cellular_automata(np.ndarray[image_t, ndim=3] data, np.ndarray[mask_t, ndim=3] mask, np.ndarray[mask_t, ndim=3] strct, np.ndarray[mask_t, ndim=3] out):

    cdef int to_return = 0
    if out is None:
        out = np.zeros_like(data)
        to_return = 1

    cdef int dx, dy, dz
    cdef int odx, ody, odz
    cdef int xo, yo, zo
    cdef int qte

    cdef image_t _min = data.min()
    cdef image_t _max = data.max()

    print("Min max", _min, _max)

    try:
        dz = data.shape[0]
        dy = data.shape[1]
        dx = data.shape[2]

        odz = strct.shape[0]
        ody = strct.shape[1]
        odx = strct.shape[2]
    except IndexError:
        dz = 1
        dy = data.shape[0]
        dx = data.shape[1]

        odz = 1
        ody = strct.shape[1]
        odx = strct.shape[2]

    cdef float[:] weights0 = np.zeros(shape=(dx * dy * dz), dtype='float32')
    cdef float[:] weights1 = np.zeros(shape=(dx * dy * dz), dtype='float32')
    cdef float **mweights = [&weights0[0], &weights1[0]]

    cdef image_t[:] _data = data.ravel()
    cdef mask_t[:] _mask = mask.ravel()
    cdef mask_t[:] _strct = strct.ravel()
    cdef mask_t[:] _out = out.ravel()

    memcpy(&_out[0], &_mask[0], mask.size)

    qte = _growcut_cellular_automata(_data, _mask, dx, dy, dz, _strct, odx, ody, odz, _min, _max, mweights, _out)
    print("Rodou", qte, "vezes")

    if to_return:
        return out
