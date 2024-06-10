# -------------------------------------------------- #
# Active Learning routines
# for the Active Learning Classifier
#
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Union, Callable

from activeclf.acquisition import pointSampler

# --- Single cycle funcion

def active_learning_cycle(feature_space: Tuple[pd.DataFrame, np.ndarray], 
                          idxs: List[int], 
                          new_batch: int, 
                          clfModel: Callable, 
                          acquisitionFunc: Callable,
                          screeningSelection: str='FPS') -> List[int]:
    
    # Extract the frature space into variables (X) and taget (y)
    X, y = feature_space

    # compute the unkown idx
    # -> the pool from where the acquisition is made
    unknown_idxs = [i for i in range(len(X)) if i not in idxs]

    # fit the clf model to the existing feature space
    clfModel.fit(X=X, y=y, idxs=idxs)
    # -> estimation of the pdf for the data
    _pdf = clfModel.predict_proba(X=X)
    # -> minmax scaling for easier value picking
    #    less noisy probability space (and entropy)
    pdf = MinMaxScaler().fit_transform(X=_pdf)

    # acquire the new data points based on the acquisition strategy
    # -> the pdf fed refers only at the unknown idxs
    if acquisitionFunc.mode == 'random':
        screen_points = acquisitionFunc.acquire(idxs=unknown_idxs, n=new_batch)
    else:
        _screen_points = acquisitionFunc.acquire(pdf=pdf[unknown_idxs], n=new_batch)
    # -> restore indexes to the POOL dataframe
        screen_points = [unknown_idxs[sc] for sc in _screen_points]

    # -> check for similar points and select only `new_batch` amount
    #    using some rule.
    #    If many points shares similar Entropy they will be picked
    #    resulting in more than `new_batch` amount.
    if len(screen_points) > new_batch and screeningSelection is not None:
        print(f"Found {len(screen_points)} points that shares the same acquisition criteria.")
        print(f"Selecting {new_batch} by '{screeningSelection}' sampling.")
        # exception for non-distance based selection (e.g., random)
        # where X is the just the indexes of the points
        if screeningSelection == 'random':
            sampled_points = pointSampler(mode=screeningSelection).sample(X=screen_points, 
                                                                          n=new_batch, 
                                                                          seed=None)
        # if the sampling is distance based X are the real points of the POOL dataframe
        # `.iloc[list[int]]` preserve the `list[int]` ordering!
        else:
            sampled_points = pointSampler(mode=screeningSelection).sample(X=X.iloc[screen_points], 
                                                                          n=new_batch)
        new_points = [screen_points[sp] for sp in sampled_points]

    else:
        new_points = screen_points

    return new_points


def get_starting_batch(data: np.ndarray, init_batch: Union[int, str]) -> List[int]:
    
    if init_batch == 'all':
        return range(0, len(data))
    elif isinstance(init_batch, int):
        return list(random.sample(range(0, len(data)), init_batch))
    else:
        raise ValueError("Only numeric values or the 'all' string is accepted.")
    
