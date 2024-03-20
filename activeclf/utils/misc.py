# -------------------------------------------------- #
# Misc / Utilities
# for the Active Learning Classifier
#
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import numpy as np
import pandas as pd
from typing import *

# --- SPACE BUILDER ---#

def get_space_lims(coord: np.ndarray, incr: Union[str, float] =None) -> Tuple[float, float]:

    coord_min, coord_max = coord.min(), coord.max()

    try:
        if isinstance(incr, str):
            try:
                _incr = int(incr.replace("%", ""))
            except:
                raise ValueError("String has to be a number and a percentage (ex. '10%')")
            tot_range = abs(coord_max - coord_min)
            incr = (tot_range/100)*_incr
            coord_min, coord_max = coord_min - incr, coord_max + incr

        elif isinstance(incr, int):
            coord_min, coord_max = coord_min - incr, coord_max + incr

        elif not incr:
            pass

    except ValueError as error:
        print(repr(error))

    return coord_min, coord_max


def make_meshgrid(x: np.ndarray, y: np.ndarray, delta: float =.2, incr='20%') -> Tuple[np.ndarray, np.ndarray]:

    x_min, x_max = get_space_lims(coord=x, incr=incr)
    y_min, y_max = get_space_lims(coord=y, incr=incr)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, delta),
                         np.arange(y_min, y_max, delta))
    
    return xx, yy

# --- ///////////// ---#
