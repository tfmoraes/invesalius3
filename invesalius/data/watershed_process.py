import numpy as np
from scipy import ndimage
from scipy.ndimage import watershed_ift, generate_binary_structure
from skimage.morphology import watershed
from invesalius.data import growcut

def get_LUT_value(data, window, level):
    shape = data.shape
    data_ = data.ravel()
    data = np.piecewise(data_,
                        [data_ <= (level - 0.5 - (window-1)/2),
                         data_ > (level - 0.5 + (window-1)/2)],
                        [0, window, lambda data_: ((data_ - (level - 0.5))/(window-1) + 0.5)*(window)])
    data.shape = shape
    return data


def do_watershed(image, markers,  tfile, shape, bstruct, algorithm, mg_size, use_ww_wl, wl, ww, q):
    mask = np.memmap(tfile, shape=shape, dtype='uint8', mode='r+')
    print("algorithm", algorithm)
    if use_ww_wl:
        if algorithm == 'Watershed':
            tmp_image = ndimage.morphological_gradient(
                           get_LUT_value(image, ww, wl).astype('uint16'),
                           mg_size)
            tmp_mask = watershed(tmp_image, markers.astype('int16'), bstruct)
        elif algorithm == 'growcut':
            print("aqui, cara!")
            tmp_image = ndimage.morphological_gradient(
                           get_LUT_value(image, ww, wl).astype('uint16'),
                           mg_size).astype('int16')
            tmp_mask = np.zeros_like(mask)
            growcut.growcut_cellular_automata(tmp_image,
                                              markers,
                                              bstruct.astype('uint8'),
                                              tmp_mask)
        else:
            tmp_image = get_LUT_value(image, ww, wl).astype('uint16')
            #tmp_image = ndimage.gaussian_filter(tmp_image, self.config.mg_size)
            #tmp_image = ndimage.morphological_gradient(
                           #get_LUT_value(image, ww, wl).astype('uint16'),
                           #self.config.mg_size)
            tmp_mask = watershed_ift(tmp_image, markers.astype('int16'), bstruct)
    else:
        if algorithm == 'Watershed':
            tmp_image = ndimage.morphological_gradient((image - image.min()).astype('uint16'), mg_size)
            tmp_mask = watershed(tmp_image, markers.astype('int16'), bstruct)
        else:
            tmp_image = (image - image.min()).astype('uint16')
            #tmp_image = ndimage.gaussian_filter(tmp_image, self.config.mg_size)
            #tmp_image = ndimage.morphological_gradient((image - image.min()).astype('uint16'), self.config.mg_size)
            tmp_mask = watershed_ift(tmp_image, markers.astype('int8'), bstruct)
    mask[:] = tmp_mask
    mask.flush()
    q.put(1)
