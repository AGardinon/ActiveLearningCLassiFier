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

from typing import List, Union


class DataLoader:
    def __init__(self, file_path: str, target: str) -> None:
        self._file_path = file_path
        self.target = target
        self.constant_columns = None
        self.X = None
        self.y = None
        self.scaler = None

        print(f"Loading DataFrame {self._file_path}\n")
        try:
            self.df = pd.read_csv(file_path).drop(labels='Unnamed: 0', axis=1)
        except:
            self.df = pd.read_csv(file_path)

        self._constant_columns = get_constant_columns(df=self.df)
        # remove the target columns from the list of columns to remove
        # just a precaution for cycle0 for real experiments
        if self.target in self._constant_columns:
            self._constant_columns.remove(self.target)

        if len(self._constant_columns) > 0:
            print(f'Removing constant columns\n{self._constant_columns}\n')
            self.constant_columns = {
                key : arg for key, arg in zip(self._constant_columns,np.unique(self.df[self._constant_columns]))
            }
            self.df = remove_columns(df=self.df, key=self._constant_columns)
        elif len(self._constant_columns) == 0:
            pass


    def merge_validated_df(self, validated_df: pd.DataFrame, target: str, overwrite: bool=False):
        assert self.scaler is None, f'Warning, you should merge before creating the `feature_space`!'
        self.df = dataframe_merger(df1=self.df, 
                                   df2_to_merge=validated_df, 
                                   target_col=target)
        if overwrite:
            self.df.to_csv(self._file_path, index=False)
            print(f'Dataset updated, ow to {self._file_path}')


    def feature_space(self, scaling: bool=True):
        if self.scaler is None:
            # target definition
            self.y = self.df[self.target]

            # creating the feature space
            self.X = self.df.drop(columns=self.target)
            # feature space labels keys
            self.fspace_keys = [k for k in self.df.columns if k != self.target]

            print(f'Feature space: {self.fspace_keys},\nTarget property: {self.target}')

            if scaling:
                print('Scaling the data (StandarScaler) ...')
                self.scaler = StandardScaler()
                self.X = self.scaler.fit_transform(X=self.X)
                self.X = pd.DataFrame(data=self.X, columns=self.fspace_keys)
            else:
                print('!!! The data might not be scaled ...')
        else:
            print('Feature space already created.')
            print(f'Feature space: {self.fspace_keys},\nTarget property: {self.target}')
            print(f'Scaler used: {self.scaler}')
            

# -------------------------------------------------- #

def get_constant_columns(df: pd.DataFrame) -> List[str]:
    const_val = list()

    for k, v in df.items():
        if len(v.unique()) == 1:
            const_val.append(k)

    return const_val


def remove_columns(df: pd.DataFrame, 
                   key: Union[List[str], str]) -> pd.DataFrame:
    
    if isinstance(key, list):
        to_drop = sum([[s for s in df.columns if k in s] for k in key], [])

    elif isinstance(key, str):
        to_drop = [s for s in df.columns if key in s]

    else:
        raise TypeError("Key parameters can only be `str` or `list`.")
    
    df = df.drop(labels=to_drop, 
                 axis=1)

    return df


def dataframe_merger(df1: pd.DataFrame, 
                     df2_to_merge: pd.DataFrame, 
                     target_col: str='Phase') -> pd.DataFrame:

    column_list = df1.columns.to_list()
    column_list.remove(target_col)

    # create a dummy merger
    merged = df1.merge(df2_to_merge, 
                       on=column_list, 
                       how='left', 
                       suffixes=('', '_real'))

    # Replace the default values in df1 with the values from df2
    merged[target_col] = merged[target_col+'_real'].combine_first(merged[target_col])

    # Drop the auxiliary column
    merged.drop(columns=[target_col+'_real'], inplace=True)

    merged[target_col] = merged[target_col].astype(int)

    return merged
