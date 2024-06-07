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
        self.X = None
        self.y = None
        self.scaler = None

        print("Loading DataFrame ...\n")
        try:
            self.df = pd.read_csv(file_path).drop(labels='Unnamed: 0', axis=1)
        except:
            self.df = pd.read_csv(file_path)

        self._constant_columns = get_constant_columns(df=self.df)
        if len(self._constant_columns) > 0:
            print(f'Removing constant columns\n{self._constant_columns}\n')
            self.constant_columns = {
                key : arg for key, arg in zip(self._constant_columns,np.unique(self.df[self._constant_columns]))
            }
            self.df = remove_columns(df=self.df, key=self._constant_columns)
        elif len(self._constant_columns) == 0:
            pass

    def merge_validated_df(self, validated_df: pd.DataFrame, target: str):
        assert self.scaler is None, f'Warning, you should merge before creating the `feature_space`!'
        self.df = dataframe_merger(df1=self.df, df2=validated_df, target_col=target)
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
            print('Scaling the data (StandarScaler) ...')
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X=self.X)
            self.X = pd.DataFrame(data=self.X, columns=self.fspace_keys)
        else:
            print('!!! The data might not be scaled ...')
            pass

# -------------------------------------------------- #

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


def dataframe_merger(df1: pd.DataFrame, 
                     df2: pd.DataFrame, 
                     target_col: str='Phase', 
                     keep: str='first') -> pd.DataFrame:
    # stack the two df
    merged_df = pd.concat([df1, df2]).drop_duplicates(keep=keep).reset_index(drop=True)

    # remove the target columns, as it pollutes the duplicates
    target_column_df = merged_df[target_col]
    merged_df = merged_df.drop(columns=target_col, axis=1)

    # list of True: dup, False: uniq
    duplicates_list = merged_df.duplicated(keep='first')
    duplicates_indxs_list = np.arange(len(target_column_df))[duplicates_list]

    # add the target column back
    merged_df[target_col] = target_column_df

    return merged_df.drop(index=duplicates_indxs_list).reset_index(drop=True)