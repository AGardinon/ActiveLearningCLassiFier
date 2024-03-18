# -------------------------------------------------- #
# Classification handler
# for the Active Learning Classifier
#
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC


class ClassifierModel:
    def __init__(self, model: str, **kwds) -> None:

        self.classification_models = {
            'SVC' : SVC,
            'GaussianProcessClassifier' : GaussianProcessClassifier,
        }

        assert model in self.classification_models.keys(), f"Specified 'method' not available. " \
                                                           f"Select from: {self.acquisition_method.keys()}"
        
        self.model = model
        self.params = kwds
        self.clf = self.classification_models[model](**kwds)


    def fit(self, X: pd.DataFrame, y: pd.DataFrame, idxs: list[int]) -> None:

        if idxs:
            self.clf.fit(X.iloc[idxs], y.iloc[idxs])
        else:
            self.clf.fit(X, y)

        pass
    

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        try:
            return self.clf.predict_proba(X=X)
        except:
            raise ValueError('Error')