# -------------------------------------------------- #
# Data Manager
# for the Active Learning Classifier
#
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, df: pd.DataFrame, **kwds) -> None:
        self.df = df
        self.params = kwds
        self.target_feature_map = None

        print("Loading DataFrame ...\n")
        self.load_data(**self.params)


    def load_data(self, target_feature: str, used_features: list[str]='all'):

        # Get target feature
        try:
            print(f"Target feature: {target_feature}")
            Y = self.df[target_feature].reset_index(drop=True)
            if Y.dtype != int:
                print("Changing categorical values to classes.")
                self.target_feature_map = {
                    v:k for k,v in enumerate(np.unique(Y.values))
                }
                print(f"Categories mapped to {self.target_feature_map}\n")
                self.Y = Y.map(self.target_feature_map)
            else:
                self.Y = Y
        except:
            raise ValueError(f"A valid `target_feature` must be selected from:\n{self.df.info()}")
        
        # Select working features
        try:
            if isinstance(used_features, list):
                self.X = self.df[used_features].reset_index(drop=True)
            elif used_features == 'all':
                self.X = self.df.drop(columns=target_feature, axis=1)
        except:
            raise ValueError(f"Working features must be selected from the ones \
                             existing in the DF (deafault 'all' uses all features)")
        print(f"Working DataFrame: {self.X.info()}")
        
        return self

    # splits the data into train and test
    def data_splits(self, test_size: float, random_state: int=73) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y,
                                                            test_size=test_size,
                                                            random_state=random_state)
        X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
        y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

        return X_train, X_test, y_train, y_test