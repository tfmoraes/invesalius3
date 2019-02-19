import numpy as np
cimport numpy as np
cimport cython

from collections import deque

from cython.parallel import prange
from libc.math cimport floor, ceil, fabs
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
from libcpp cimport bool
from libcpp.deque cimport deque as cdeque
from libcpp.vector cimport vector

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
cdef int _growcut_cellular_automata_queue(image_t[:] data, mask_t[:] mask, int dx, int dy, int dz, mask_t[:] strct, int odx, int ody, int odz, image_t _min, image_t _max, mask_t[:] out) nogil:
    cdef int i, j, sj, x, y, z, xo, yo, zo, sx, sy, sz
    cdef int qte = 0

    cdef int di = dx * dy * dz
    cdef int dj = odx *ody * odz

    cdef int offset_z = odz / 2
    cdef int offset_y = ody / 2
    cdef int offset_x = odx / 2

    cdef float weight = 0


    cdef int aweight = 0
    cdef int nweight = 1

    cdef cdeque[int] queue

    cdef vector[float] mweights
    mweights.reserve(di)

    cdef vector[bool] included
    included.reserve(di)

    for i in range(di):
        if mask[i]:
            mweights[i] = 1

            z = i / (dx * dy)
            y = (i / dx) % dy
            x = i % dx

            for j in range(dj):
                if strct[j]:
                    sz = j / (odx * ody)
                    sy = (j / odx) % ody
                    sx = j % (odx)

                    zo = z + sz - offset_z
                    yo = y + sy - offset_y
                    xo = x + sx - offset_x

                    if 0 <= xo < dx and 0 <= yo < dy and 0 <= zo < dz:
                        sj = zo * dx * dy + yo * dx + xo
                        if mask[sj] == 0 and not included[i]:
                            queue.push_back(i)
                            included[i] = True
                            break
        else:
            mweights[i] = 0

    while queue.size():
        qte += 1
        i = queue.front()
        queue.pop_front()
        included[i] = False
        z = i / (dx * dy)
        y = (i / dx) % dy
        x = i % dx

        for j in range(dj):
            if strct[j]:
                sz = j / (odx * ody)
                sy = (j / odx) % ody
                sx = j % (odx)

                zo = z + sz - offset_z
                yo = y + sy - offset_y
                xo = x + sx - offset_x

                sj = zo * dx * dy + yo * dx + xo
                if 0 <= xo < dx and 0 <= yo < dy and 0 <= zo < dz and mweights[sj] < 1.0:
                    weight = g(fabs(data[i] - data[sj]), _min, _max) * mweights[i]

                    if weight > mweights[sj]:
                        out[sj] = out[i]
                        mweights[sj] = weight

                        if not included[sj]:
                            queue.push_back(sj)

    return qte



@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def growcut_cellular_automata(np.ndarray[image_t, ndim=1] data, np.ndarray[mask_t, ndim=1] mask, int dx, int dy, int dz, np.ndarray[mask_t, ndim=1] strct, int odx, int ody, int odz, image_t _min, image_t _max, np.ndarray[mask_t, ndim=1] out):
    cdef int qte
    print("Min max", _min, _max)
    print("dx, dy, dz", dx, dy, dz)

    cdef image_t[:] _data = data.ravel()
    cdef mask_t[:] _mask = mask.ravel()
    cdef mask_t[:] _strct = strct.ravel()
    cdef mask_t[:] _out = out.ravel()

    print("memcpy")
    memcpy(&_out[0], &_mask[0], mask.size)

    print("vou rodar")
    qte = _growcut_cellular_automata_queue(_data, _mask, dx, dy, dz, _strct, odx, ody, odz, _min, _max, _out)
    print("Rodou", qte, "vezes")
