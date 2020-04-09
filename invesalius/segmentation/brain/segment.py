import itertools
import os
import pathlib
import sys

import numpy as np
from skimage.transform import resize

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))

from invesalius import inv_paths
from invesalius.segmentation.brain import utils


SIZE = 48
OVERLAP = SIZE // 2 + 1


def image_normalize(image, min_=0.0, max_=1.0):
    imin, imax = image.min(), image.max()
    return (image - imin) * ((max_ - min_) / (imax - imin)) + min_


def gen_patches(image, patch_size, overlap):
    sz, sy, sx = image.shape
    i_cuts = list(
        itertools.product(
            range(0, sz, patch_size - OVERLAP),
            range(0, sy, patch_size - OVERLAP),
            range(0, sx, patch_size - OVERLAP),
        )
    )
    sub_image = np.empty(shape=(patch_size, patch_size, patch_size), dtype="float32")
    for idx, (iz, iy, ix) in enumerate(i_cuts):
        sub_image[:] = 0
        _sub_image = image[
            iz : iz + patch_size, iy : iy + patch_size, ix : ix + patch_size
        ]
        sz, sy, sx = _sub_image.shape
        sub_image[0:sz, 0:sy, 0:sx] = _sub_image
        ez = iz + sz
        ey = iy + sy
        ex = ix + sx

        yield (idx + 1.0) / len(i_cuts), sub_image, ((iz, ez), (iy, ey), (ix, ex))


def predict_patch(sub_image, patch, nn_model, patch_size=SIZE):
    (iz, ez), (iy, ey), (ix, ex) = patch
    sub_mask = nn_model.predict(
        sub_image.reshape(1, patch_size, patch_size, patch_size, 1)
    )
    return sub_mask.reshape(patch_size, patch_size, patch_size)[
        0 : ez - iz, 0 : ey - iy, 0 : ex - ix
    ]


def brain_segment(image, probability_array, comm_array):
    import keras

    # Loading model
    folder = inv_paths.MODELS_DIR.joinpath("brain_mri_t1")
    with open(folder.joinpath("model.json"), "r") as json_file:
        model = keras.models.model_from_json(json_file.read())
    model.load_weights(str(folder.joinpath("model.h5")))
    model.compile("Adam", "binary_crossentropy")

    image = image_normalize(image, 0.0, 1.0)
    sums = np.zeros_like(image)
    # segmenting by patches
    for completion, sub_image, patch in gen_patches(image, SIZE, OVERLAP):
        comm_array[0] = completion
        print("completion", completion)
        (iz, ez), (iy, ey), (ix, ex) = patch
        sub_mask = predict_patch(sub_image, patch, model, SIZE)
        probability_array[iz:ez, iy:ey, ix:ex] += sub_mask
        sums[iz:ez, iy:ey, ix:ex] += 1

    probability_array /= sums
    comm_array[0] = np.Inf


def main():
    image_filename = sys.argv[1]
    image_dtype = sys.argv[2]
    sz = int(sys.argv[3])
    sy = int(sys.argv[4])
    sx = int(sys.argv[5])
    prob_arr_filename = sys.argv[6]
    comm_arr_filename = sys.argv[7]
    backend = sys.argv[8]
    device_id = sys.argv[9]
    use_gpu = bool(int(sys.argv[10]))

    image = np.memmap(image_filename, dtype=image_dtype, shape=(sz, sy, sx), mode="r")
    probability_array = np.memmap(
        prob_arr_filename, dtype=np.float32, shape=(sz, sy, sx), mode="r+"
    )
    comm_array = np.memmap(comm_arr_filename, dtype=np.float32, shape=(1,), mode="r+")

    utils.prepare_ambient(backend, device_id, use_gpu)
    brain_segment(image, probability_array, comm_array)


if __name__ == "__main__":
    main()
