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

    cdef cdeque[int] queue

    cdef vector[float] mweights
    mweights.reserve(di)

    with gil:
        print("preenchendo weights")
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
                        if mask[sj] == 0:
                            queue.push_back(i)
                            break
        else:
            mweights[i] = 0

    with gil:
        print("workgin")
    while queue.size():
        qte += 1
        i = queue.front()
        queue.pop_front()
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

                        queue.push_back(sj)

    return qte


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
cdef int _growcut_cellular_automata(image_t[:] data, mask_t[:] mask, int dx, int dy, int dz, mask_t[:] strct, int odx, int ody, int odz, image_t _min, image_t _max, mask_t[:] out) nogil:
    cdef int i, j, sj, x, y, z, xo, yo, zo, sx, sy, sz
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

    cdef vector[float] mweights
    mweights.reserve(di * 2)

    for i in prange(di):
        if mask[i]:
            mweights[0*di + i] = 1
            mweights[1*di + i] = 1
        else:
            mweights[0*di + i] = 0
            mweights[1*di + i] = 0

    while modified:
        modified = 0
        for i in prange(di):
            vmodified = 0
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
                        weight = g(fabs(data[i] - data[sj]), _min, _max) * mweights[aweight * di + sj]

                        if weight > mweights[aweight * di + i]:
                            out[i] = out[sj]
                            mweights[nweight*di + i] = weight
                            modified += 1
                            vmodified = 1
            if not vmodified:
                mweights[nweight*di + i] = mweights[aweight*di + i]

        aweight = nweight
        nweight = (nweight + 1) % 2
        qte += 1

    return qte


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False)
def growcut_cellular_automata(np.ndarray[image_t, ndim=1] data, np.ndarray[mask_t, ndim=1] mask, np.ndarray[mask_t, ndim=1] strct, np.ndarray[mask_t, ndim=1] out):
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
