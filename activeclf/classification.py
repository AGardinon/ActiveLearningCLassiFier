# -------------------------------------------------- #
# Classification handler
# for the Active Learning Classifier
#
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import numpy as np
import pandas as pd
from typing import List

from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


class ClassifierModel:
    def __init__(self, model: str, **kwds) -> None:

        self.classification_models = {
            'SVC' : SVC,
            'GaussianProcessClassifier' : GaussianProcessClassifier,
            'NaiveBayes' : GaussianNB,
            'RandomForest' : RandomForestClassifier,
            'AdaBoost' : AdaBoostClassifier,
            'MLPC' : MLPClassifier,
            'GBoost' : GradientBoostingClassifier
        }

        assert model in self.classification_models.keys(), f"Specified 'method' not available. " \
                                                           f"Select from: {list(self.classification_models.keys())}"
        
        self.model = model
        self.params = kwds
        self.clf = self.classification_models[model](**kwds)


    def fit(self, X: pd.DataFrame, y: pd.DataFrame, idxs: List[int]) -> None:
        # revert back to the original clf after using the SingleClassGaussianModel
        if self.clf.__class__.__name__ == SingleClassGaussianModel.__name__:
            self.clf = self.classification_models[self.model](**self.params)

        if idxs:
            try:
                self.clf.fit(X.iloc[idxs], y.iloc[idxs])
            except:
                print('Warning, only one class!!')
                print('A simple Gaussian Model will be used to fit '\
                      f'the data instead of the selected classifier ({self.model}).')
                self.clf = SingleClassGaussianModel(n_dim=X.shape[1])
                self.clf.fit(mean=X.iloc[idxs], cov='eye')
        else:
            self.clf.fit(X, y)
    

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        try:
            return self.clf.predict_proba(X=X)
        except:
            raise ValueError('Error, the selected classifier does \
                             not have a .predict_proba() attribute.')
        

class SingleClassGaussianModel:
    def __init__(self, n_dim: int) -> None:
        self.n_dim = n_dim
        self._is_fit = False
        self.Z = None
        self.mean = None
        self.cov = None


    def __repr__(self):
        return self.__class__.__name__
    

    def fit(self, mean: List[np.ndarray], cov: str='eye') -> None:
        if isinstance(mean, pd.DataFrame):
            self.mean = mean.to_numpy()
        else:
            self.mean = mean
        if cov == 'eye':
            self.cov = np.eye(N=self.n_dim)
        else:
            self.cov = cov
        self._is_fit = True


    def predict(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.ones(shape=(len(X),),dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        densities_list = list()
        for centre in self.mean:
            densities_list.append(
                [multivariate_gaussian(x=p, mean=centre, cov=self.cov) for p in X]
            )
        self.Z = np.array([sum(z) for z in zip(*densities_list)])
        return self.Z


def multivariate_gaussian(x: np.ndarray, 
                          mean: np.ndarray, 
                          cov: np.ndarray) -> float:
    # dimensions are inferred from the
    # shape of the mean value
    k = mean.shape[0]
    x_m = x - mean
    return np.exp(-0.5 * np.dot(x_m, np.linalg.solve(cov, x_m))) / \
           np.sqrt((2 * np.pi) ** k * np.linalg.det(cov))