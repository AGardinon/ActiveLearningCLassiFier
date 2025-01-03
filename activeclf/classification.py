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



from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

'''
KernelFactoru usage.

eample 1:
classifier_dict = {
    'kernel': ['*', {'type': 'C', 'constant_value': 1.0}, {'type': 'RBF', 'length_scale': 1.0}],
    'n_restarts_optimizer': 5,
    'max_iter_predict': 150,
    'n_jobs': 3
}

kernel_factory = KernelFactory(kernel_dict=classifier_dict['kernel'])
kernel = kernel_factory.get_kernel()

-> kernel = 1**2 * RBF(length_scale=1)

classifier_dict = {
    'kernel': ['+', {'type': 'C', 'constant_value': 2.0}, ['*', 'RBF', {'type': 'W', 'noise_level': 1.0}]],
    'n_restarts_optimizer': 5,
    'max_iter_predict': 150,
    'n_jobs': 3
}

kernel_factory = KernelFactory(kernel_dict=classifier_dict['kernel'])
kernel = kernel_factory.get_kernel()

-> kernel = 1.41**2 + RBF(length_scale=1) * WhiteKernel(noise_level=1)

'''

class KernelFactory:
    def __init__(self, kernel_dict: dict):
        self.kernel_dict = kernel_dict

    def get_kernel(self):
        kernel_map = {
            'RBF': RBF,
            'Matern': Matern,
            'RationalQuadratic': RationalQuadratic,
            'C': ConstantKernel,
            'W': WhiteKernel,
        }
        
        return self._parse_kernel(self.kernel_dict, kernel_map)
    
    def _parse_kernel(self, kernel_dict, kernel_map):
        # Simple string case
        if isinstance(kernel_dict, str):
            return kernel_map[kernel_dict]()

        # Dictionary with parameters
        elif isinstance(kernel_dict, dict):
            kernel_type = kernel_dict.pop('type')
            return kernel_map[kernel_type](**kernel_dict)

        # Composite kernel
        elif isinstance(kernel_dict, list):
            operator = kernel_dict[0]
            first_kernel = self._parse_kernel(kernel_dict[1], kernel_map)
            second_kernel = self._parse_kernel(kernel_dict[2], kernel_map)
            
            if operator == '*':
                return first_kernel * second_kernel
            elif operator == '+':
                return first_kernel + second_kernel
            else:
                raise ValueError(f"Unknown operator: {operator}")

        else:
            raise ValueError(f"Invalid kernel definition: {kernel_dict}")