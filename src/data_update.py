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


def label_encoding(df):
    """
    Convert the dataframe features into one hot encoding matrix

    Args:
        df_train: training dataframe
        df_valid: validation dataframe

    Returns:
    """
    features = [i for i in df.columns if i not in ['income', 'kfold']]
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df[col] = lbl.transform(df[col])
    """ use transform when we have already fit it,use fit_transform which 
        is a combination of fit and transform together in 1 api, in some 
        situations you only want to use training data to learn model 
        parameters and apply the same in test data set also. """

    return df