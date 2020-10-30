"""
This script is the main file that calls all the other
scripts to run the ml project.
"""

import os
import joblib
import argparse
import pandas as pd
import src.config as sc

from sklearn import metrics
from src.create_folds import create_folds_using_kfold
from src.model_dispatcher import model
from src.data_update import fill_na_with_none, one_hot_encoding


def run_output(fold, df):
    """
    Structure, train and save the model
    for given fold number.

    Args:
        fold (int): number for fold
        df (pd.DataFrame): training dataset

    Returns:

    """
    df_new = fill_na_with_none(df)
    df_train = df_new[df_new['kfold'] != fold].reset_index(drop=True)
    df_valid = df_new[df_new['kfold'] == fold].reset_index(drop=True)

    """Apply one hot encoding to feature matrix"""
    x_train, x_valid = one_hot_encoding(df_train, df_valid)

    """Convert training and validation dataframe target to numpy values for AUC calculation"""
    y_train = df_train['target'].values
    y_valid = df_valid['target'].values

    """import the model required"""
    clf = model

    """fit model on the training data"""
    clf.fit(x_train, y_train)

    """As target variable is skewed we will need predicted probabilities to calculate AUC score"""
    y_pred = clf.predict_proba(x_valid)[:,1]

    """find accuracy as distribution of all target variables in similar"""
    auc = metrics.roc_auc_score(y_valid, y_pred)
    print(f"Fold number :{fold}, AUC score : {auc}")

    """Save Model"""
    joblib.dump(clf, os.path.join(sc.OUTPUT_FILE, f'dt_{fold}.bin'))


if __name__ == '__main__':
    df = create_folds_using_kfold()
    """Create a parser object and add variables that you want to declare"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    """We have used LR as a model but we can also use Random forest tree.
        For that we can use label encoding instead of one hot encoding
        and You will observe that RF without tuning of hyper parameters 
        performs a lot worse the simple LR. So it is better to always 
        start with the simple model. Other parameters that might influence 
        model decision is time taken in computation. RF takes much longer time.
        Even on sparse one hot encoded data, RF will take more time.
        Also better choice is to use label encoding for tree based algorithms even 
        XG boost or other similar gradient boosting algorithms
        """
    run_output(args.fold, df)
