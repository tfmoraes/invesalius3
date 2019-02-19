import numpy as np
from invesalius.data import _growcut


def growcut_cellular_automata(data, mask, strct, out=None):
    to_return = 0
    if out is None:
        out = np.zeros_like(data)
        to_return = 1

    _growcut._growcut_cellular_automata_queue(
        data.ravel(), mask.ravel(), strct.ravel(), out.ravel()
    )

    if to_return:
        return out
