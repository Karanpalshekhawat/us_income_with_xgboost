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
from src.data_update import fill_na_with_none, label_encoding


def run_output(fold, df):
    """
    Structure, train and save the model
    for given fold number.

    Args:
        fold (int): number for fold
        df (pd.DataFrame): training dataset

    Returns:

    """
    numerical_columns = ['fnlwgt', 'age', 'capital.gain', 'capital.loss', 'hours.per.week']
    df = df.drop(numerical_columns, axis=1)

    """Map income target to numerical 0s and 1s"""
    target_mapping = {'<=50K': 0, '>50K': 1}
    df['income'] = df['income'].apply(lambda x: target_mapping[x])

    df_new = fill_na_with_none(df)

    """ Apply label encoding to feature matrix,
        it means that we will convert categories 
        in each feature columns to some number
    """
    df_new = label_encoding(df_new)

    df_train = df_new[df_new['kfold'] != fold].reset_index(drop=True)
    df_valid = df_new[df_new['kfold'] == fold].reset_index(drop=True)

    """Convert training and validation dataframe to numpy values for AUC calculation"""

    features = [i for i in df.columns if i not in ['income', 'kfold']]
    x_train = df_train[features].values
    x_valid = df_valid[features].values
    y_train = df_train['income'].values
    y_valid = df_valid['income'].values

    """import the model required"""
    clf = model

    """fit model on the training data"""
    clf.fit(x_train, y_train)

    """As target variable is skewed we will need predicted probabilities to calculate AUC score"""
    y_pred = clf.predict_proba(x_valid)[:, 1]

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
    """ We are using label encoding and xgboost algo, 
        note that we can also us LR with one hot encoding, it
        may give more accurate results.
    """
    run_output(args.fold, df)
