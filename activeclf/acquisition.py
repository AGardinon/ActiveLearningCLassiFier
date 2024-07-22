# -------------------------------------------------- #
# Acquisition functions
# for the Active Learning Classifier
#
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

# --- ACQUISITION FUNCTIONS ---#

import scipy
import numpy as np
from typing import List


class DecisionFunction:
    def __init__(self, mode: str, seed: int=None, decimals: int=2) -> None:
        
        self.decision_modes = {
            'exploitation' : exploitation,
            'exploration' : exploration,
            'random' : self._random_pick
        }

        assert mode in self.decision_modes.keys(), f"Specified 'mode' not available. " \
                                                   f"Select from: {self.decision_modes.keys()}"
        
        self.mode = mode
        self.rng = np.random.default_rng(seed=seed)
        self.decimals = decimals

    def acquire(self, **kwrgs):
        if self.mode == 'random':
            return self._random_pick(**kwrgs)
        else:
            return self._acquire(**kwrgs)
        

    def _acquire(self, pdf: np.ndarray, n: int=1) -> List[int]:

        return self.decision_modes[self.mode](pdf=pdf, n=n, decimals=self.decimals)


    def _random_pick(self, idxs: List[int], n: int=1) -> List[int]:

        rng_idx_pick = self.rng.integers(0, len(idxs), n)
        return [idxs[r] for r in rng_idx_pick]
    


def exploitation(pdf: np.ndarray, n: int=1, decimals: int=2) -> List[int]:

    entropy = np.around(scipy.stats.entropy(pdf, axis=1), decimals=decimals)
    new_points = np.argsort(entropy)[:n]
    new_idxs = np.concatenate([[i for i,val in enumerate(entropy) if val == entropy[h]] 
                               for h in new_points])

    return list(set(new_idxs))


def exploration(pdf: np.ndarray, n: int=1, decimals: int=2) -> List[int]:
    
    entropy = np.around(scipy.stats.entropy(pdf, axis=1), decimals=decimals)
    new_points = np.argsort(entropy)[::-1][:n]
    new_idxs = np.concatenate([[i for i,val in enumerate(entropy) if val == entropy[h]] 
                               for h in new_points])

    return list(set(new_idxs))

# --- ///////////////////// ---#


# --- POINTS SAMPLER ---#

import random
import pandas as pd
from typing import Union


class pointSampler:
    def __init__(self, mode: str, seed: int=None) -> None:
        
        self.sampling_modes = {
            'random' : sampling_rand,
            'FPS' : sampling_fps
        }

        assert mode in self.sampling_modes.keys(), f"Specified 'mode' not available. " \
                                                   f"Select from: {self.sampling_modes.keys()}"
        
        self.mode = mode

    def sample(self, X: Union[List[int], np.ndarray], n: int, **kwrgs) -> List[int]:
        
        return self.sampling_modes[self.mode](X=X, n=n, **kwrgs)

    
def sampling_rand(X: List[int], n: int, seed: int=73) -> List[int]:

    rng_idx_pick = random.sample(range(0, len(X)), n)
    
    return rng_idx_pick


def sampling_fps(X: np.ndarray, n: int, start_idx: int=None, 
                return_distD: bool=False) -> Union[List[int], np.ndarray]:

    if isinstance(X, pd.DataFrame):
        X = np.array(X)

    # init the output quantities
    fps_ndxs = np.zeros(n, dtype=int)
    distD = np.zeros(n)

    # check for starting index
    if not start_idx:
        # the b limits has to be decreaed because of python indexing
        # start from zero
        start_idx = random.randint(a=0, b=X.shape[0]-1)
    # inset the first idx of the sampling method
    fps_ndxs[0] = start_idx

    # compute the distance from selected point vs all the others
    dist1 = np.linalg.norm(X - X[start_idx], axis=1)

    # loop over the distances from selected starter
    # to find the other n points
    for i in range(1, n):
        # get and store the index for the max dist from the point chosen
        fps_ndxs[i] = np.argmax(dist1)
        distD[i - 1] = np.amax(dist1)

        # compute the dists from the newly selected point
        dist2 = np.linalg.norm(X - X[fps_ndxs[i]], axis=1)
        # takes the min from the two arrays dist1 2
        dist1 = np.minimum(dist1, dist2)

        # little stopping condition
        if np.abs(dist1).max() == 0.0:
            print(f"Only {i} iteration possible")
            return fps_ndxs[:i], distD[:i]
        
    if return_distD:
        return list(fps_ndxs), distD
    else:
        return list(fps_ndxs)
    
# --- ////////////// ---#