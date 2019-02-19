import numpy as np
from invesalius.data import _growcut
from invesalius.utils import timing


@timing
def growcut_cellular_automata(data, mask, strct, out=None):
    data = np.ascontiguousarray(data)
    mask = np.ascontiguousarray(mask)
    strct = np.ascontiguousarray(strct)

    to_return = 0
    if out is None:
        out = np.zeros_like(mask)
        to_return = 1

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
        ody = strct.shape[0]
        odx = strct.shape[1]

    _min = data.min()
    _max = data.max()

    print("types", type(data), type(mask), type(strct), type(out))

    _growcut.growcut_cellular_automata(
        data.ravel(),
        mask.ravel(),
        dx,
        dy,
        dz,
        strct.ravel(),
        odx,
        ody,
        odz,
        _min,
        _max,
        out.ravel(),
    )

    print("TERMINEI")

    if to_return:
        return out
