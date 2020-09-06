import cython

@cython.boundscheck(False)
cpdef unsigned short[:, :] substitute_distant_pixels(unsigned short [:, :] image, depth_frame):
    cdef int x, y, w, h
    h = image.shape[0]
    w = image.shape[1]

    for y in range(0, h):
        for x in range(0, w):
            image[y, x] = 9000 if image[y, x] > 1000 else image[y, x]

    return image