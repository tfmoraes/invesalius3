import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport floor, ceil, sqrt, fabs, round
from cython.parallel import prange

DTYPE8 = np.uint8
ctypedef np.uint8_t DTYPE8_t

DTYPEF64 = np.float64
ctypedef np.float64_t DTYPEF64_t



@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef inline DTYPEF64_t GS(DTYPEF64_t[:, :, :] I, int z, int y, int x) nogil:
    cdef int dz = I.shape[0]
    cdef int dy = I.shape[1]
    cdef int dx = I.shape[2]

    if 0 <= x < dx \
            and 0 <= y < dy \
            and 0 <= z < dz:
        return I[z, y, x]
    else:
        return 0



@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef void perim(DTYPE8_t[:, :, :] image,
                DTYPE8_t[:, :, :] out) nogil:

    cdef int dz = image.shape[0]
    cdef int dy = image.shape[1]
    cdef int dx = image.shape[2]

    cdef int z, y, x
    cdef int z_, y_, x_

    for z in prange(dz, nogil=True):
        for y in xrange(dy):
            for x in xrange(dx):
                for z_ in xrange(z-1, z+2, 2):
                    for y_ in xrange(y-1, y+2, 2):
                        for x_ in xrange(x-1, x+2, 2):
                            if 0 <= x_ < dx \
                                    and 0 <= y_ < dy \
                                    and 0 <= z_ < dz \
                                    and image[z, y, x] != image[z_, y_, x_]:
                                out[z, y, x] = 1
                                break



@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef DTYPEF64_t calculate_H(DTYPEF64_t[:, :, :] I, int z, int y, int x) nogil:
    # double fx, fy, fz, fxx, fyy, fzz, fxy, fxz, fyz, H
    cdef DTYPEF64_t fx, fy, fz, fxx, fyy, fzz, fxy, fxz, fyz, H
    # int h, k, l
    cdef int h = 1
    cdef int k = 1
    cdef int l = 1

    fx = (GS(I, z, y, x + h) - GS(I, z, y, x - h)) / (2.0*h)
    fy = (GS(I, z, y + k, x) - GS(I, z, y - k, x)) / (2.0*k)
    fz = (GS(I, z + l, y, x) - GS(I, z - l, y, x)) / (2.0*l)

    fxx = (GS(I, z, y, x + h) - 2*GS(I, z, y, x) + GS(I, z, y, x - h)) / (h*h)
    fyy = (GS(I, z, y + k, x) - 2*GS(I, z, y, x) + GS(I, z, y - k, x)) / (k*k)
    fzz = (GS(I, z + l, y, x) - 2*GS(I, z, y, x) + GS(I, z - l, y, x)) / (l*l)

    fxy = (GS(I, z, y + k, x + h) - GS(I, z, y - k, x + h) \
            - GS(I, z, y + k, x - h) + GS(I, z, y - k, x - h)) \
            / (4.0*h*k)
    fxz = (GS(I, z + l, y, x + h) - GS(I, z + l, y, x - h) \
            - GS(I, z - l, y, x + h) + GS(I, z - l, y, x - h)) \
            / (4.0*h*l)
    fyz = (GS(I, z + l, y + k, x) - GS(I, z + l, y - k, x) \
            - GS(I, z - l, y + k, x) + GS(I, z - l, y - k, x)) \
            / (4.0*k*l)

    if fx*fx + fy*fy + fz*fz > 0:
        H = ((fy*fy + fz*fz)*fxx + (fx*fx + fz*fz)*fyy \
                + (fx*fx + fy*fy)*fzz - 2*(fx*fy*fxy \
                + fx*fz*fxz + fy*fz*fyz)) \
                / (fx*fx + fy*fy + fz*fz)
    else:
        H = 0.0

    return H


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
cdef void replicate(DTYPEF64_t[:, :, :] source, DTYPEF64_t[:, :, :] dest) nogil:
    cdef int dz = source.shape[0]
    cdef int dy = source.shape[1]
    cdef int dx = source.shape[2]
    cdef int x, y, z
    for z in prange(dz, nogil=True):
        for y in xrange(dy):
            for x in xrange(dx):
                dest[z, y, x] = source[z, y, x]


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.cdivision(True)
def smooth(np.ndarray[DTYPE8_t, ndim=3] image,
           int n, int bsize,
           np.ndarray[DTYPEF64_t,  ndim=3] out):

    cdef np.ndarray[DTYPE8_t, ndim=3] mask = np.zeros_like(image)
    cdef np.ndarray[DTYPE8_t, ndim=3] _mask = np.zeros_like(image)
    cdef np.ndarray[DTYPEF64_t, ndim=3] aux = np.zeros_like(out)

    cdef int i, x, y, z, S;
    cdef DTYPEF64_t H, v, cn
    cdef DTYPEF64_t diff=0.0
    cdef DTYPEF64_t dt=1/6.0

    cdef DTYPEF64_t E = 0.001

    _mask[:] = image
    for i in xrange(bsize):
        perim(_mask, mask)
        _mask[:] = mask
        print i

    # out[:] = mask

    del _mask

    cdef int dz = image.shape[0]
    cdef int dy = image.shape[1]
    cdef int dx = image.shape[2]

    S = 0
    for z in prange(dz, nogil=True):
        for y in xrange(dy):
            for x in xrange(dx):
                if image[z, y, x]:
                    out[z, y, x] = 1.0
                else:
                    out[z, y, x] = -1.0

                if mask[z, y, x]:
                    S += 1

    for i in xrange(n):
        replicate(out, aux);
        diff = 0.0;

        for z in xrange(dz):
            for y in xrange(dy):
                for x in xrange(dx):
                    if mask[z, y, x]:
                        H = calculate_H(aux, z, y, x);
                        v = aux[z, y, x] + dt*H;

                        if image[z, y, x]:
                            if v > 0:
                                out[z, y, x] = v
                            else:
                                out[z, y, x] = 0.0001
                        else:
                            if v < 0:
                                out[z, y, x] = v
                            else:
                                out[z, y, x] = -0.0001

                        diff += (out[z, y, x] - aux[z, y, x])*(out[z, y, x] - aux[z, y, x])

        cn = sqrt((1.0/S) * diff);
        print "%d - CN: %.28f - diff: %.28f\n" % (i, cn, diff)

        if cn <= E:
            break;
