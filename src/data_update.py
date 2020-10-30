"""
This script modules are used to update dataframes like
filling the NaN values, update the data type.
Also, doing manupulation like one hot encoder.
"""

import pandas as pd
from sklearn import preprocessing

def fill_na_with_none(df):
    """
    Fill all categorical NaN values with string type None
    so that we can use sklearn label encoding package
    Args:
        df (pd.DataFrame): training dataset

    Returns:
    """
    features = [i for i in df.columns if i not in ['income', 'kfold']]
    for feat in features:
        df.loc[:, feat] = df[feat].astype(str).fillna("NONE")

    return df


def one_hot_encoding(df_train, df_valid):
    """
    Convert the dataframe features into one hot encoding matrix

    Args:
        df_train: training dataframe
        df_valid: validation dataframe

    Returns:
    """
    ohe = preprocessing.OneHotEncoder()
    features = [i for i in df_train.columns if i not in ['id', 'target', 'kfold']]
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    ohe.fit(full_data[features])

    """ use transform when we have already fit it,use fit_transform which 
        is a combination of fit and transform together in 1 api, in some 
        situations you only want to use training data to learn model 
        parameters and apply the same in test data set also. """
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    return x_train, x_valid