# -------------------------------------------------- #
# Data Manager
# for the Active Learning Classifier
#
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from typing import Union


class DataLoader:
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self.constant_columns = None

        print("Loading DataFrame ...\n")
        try:
            self.df = pd.read_csv(file_path).drop(labels='Unnamed: 0', axis=1)
        except:
            self.df = pd.read_csv(file_path).drop(labels='Unnamed: 0', axis=1)

        self._constant_columns = get_constant_columns(df=self.df)
        if len(self._constant_columns) > 0:
            print(f'Removing constant columns\n{self._constant_columns}\n')
            self.constant_columns = {
                key : arg for key, arg in zip(self._constant_columns,np.unique(self.df[self._constant_columns]))
            }
            self.df = remove_columns(df=self.df, key=self._constant_columns)
        elif len(self._constant_columns) == 0:
            pass


    def feature_space(self, target: str, scaling: bool=True):
        # target definition
        self.target = target
        self.y = self.df[self.target]

        # creating the feature space
        self.X = self.df.drop(columns=self.target)
        # feature space labels keys
        self.fspace_keys = [k for k in self.df.columns if k != self.target]

        print(f'Feature space: {self.fspace_keys},\nTarget property: {target}')

        if scaling:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X=self.X)
            self.X = pd.DataFrame(data=self.X, columns=self.fspace_keys)
        else:
            pass


def get_constant_columns(df: pd.DataFrame) -> list[str]:
    const_val = list()

    for k, v in df.items():
        if len(v.unique()) == 1:
            const_val.append(k)

    return const_val


def remove_columns(df: pd.DataFrame, 
                   key: Union[list[str], str]) -> pd.DataFrame:
    
    if isinstance(key, list):
        to_drop = sum([[s for s in df.columns if k in s] for k in key], [])

    elif isinstance(key, str):
        to_drop = [s for s in df.columns if key in s]

    else:
        raise TypeError("Key parameters can only be `str` or `list`.")
    
    df = df.drop(labels=to_drop, 
                 axis=1)

    return df

# OLD CODE
# class DataLoader:
#     def __init__(self, df: pd.DataFrame, **kwds) -> None:
#         self.df = df
#         self.params = kwds
#         self.target_feature_map = None

#         print("Loading DataFrame ...\n")
#         self.load_data(**self.params)


#     def load_data(self, target_feature: str, used_features: list[str]='all'):

#         # Get target feature
#         try:
#             print(f"Target feature: {target_feature}")
#             Y = self.df[target_feature].reset_index(drop=True)
#             if Y.dtype != int:
#                 print("Changing categorical values to classes.")
#                 self.target_feature_map = {
#                     v:k for k,v in enumerate(np.unique(Y.values))
#                 }
#                 print(f"Categories mapped to {self.target_feature_map}\n")
#                 self.Y = Y.map(self.target_feature_map)
#             else:
#                 self.Y = Y
#         except:
#             raise ValueError(f"A valid `target_feature` must be selected from:\n{self.df.info()}")
        
#         # Select working features
#         try:
#             if isinstance(used_features, list):
#                 self.X = self.df[used_features].reset_index(drop=True)
#             elif used_features == 'all':
#                 self.X = self.df.drop(columns=target_feature, axis=1)
#         except:
#             raise ValueError(f"Working features must be selected from the ones \
#                              existing in the DF (deafault 'all' uses all features)")
#         print(f"Working DataFrame: {self.X.info()}")
        
#         return self

#     # splits the data into train and test
#     def data_splits(self, test_size: float, random_state: int=73) -> \
#         Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:

#         X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y,
#                                                             test_size=test_size,
#                                                             random_state=random_state)
#         X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
#         y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

#         return X_train, X_test, y_train, y_test