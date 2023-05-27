import numpy as np
import pandas as pd
import random
import math
from collections import Counter


class Utils:
    @staticmethod
    def normalize(df):
        for name in df.columns:
            if np.issubdtype(df[name].dtype, np.number):
                df[name] = (df[name] - df[name].min())/(df[name].max() - df[name].min())
        return df

    @staticmethod
    def standardize(df,target):
        for name in df.drop(target,axis=1).columns:
            df[name] = (df[name] - df[name].mean())/df[name].std()
        return df
    @staticmethod
    def split(df, size):
        train_size = int(size * len(df))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        return train_df, test_df

    @staticmethod
    def shuffle(data):
        length = len(data)
        for x in range(1,length):
            i = random.randint(0, length-x)
            data.iloc[i], data.iloc[-1-x] = data.iloc[-1-x], data.iloc[i]
        return data

    @staticmethod
    def distance(x, y, m=1):
        dist = (sum([pow(abs(a-b), m) for a, b in zip(x, y)])) ** (1/m)
        return dist





#%%
