"""
This script is use to create folds in the training set
and train the input data for some particular fold
"""

import pandas as pd
import src.config as sc

from sklearn import model_selection


def create_folds_using_kfold():
    df = pd.read_csv(sc.TRAINING_FILE)
    df['Kfold'] = -1
    """Sampling the dataframe"""
    df = df.sample(frac=1).reset_index(drop=True)
    y = df['income'].values
    """As the target distribution is skewed, better to use stratified fold 
    as it will maintain the ratio of positive to negative exaples"""
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(df, y)):
        df.loc[v_, 'kfold'] = f

    return df

