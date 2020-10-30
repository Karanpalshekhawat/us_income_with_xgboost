"""
This script plots the target distribution
to check whether which evaluation metric to be used
either accuracy, F1 score, if distribution is skewed
we should use stratified k fold and AUC to determine model
"""

import seaborn as sns


def plot_target(df):
    b = sns.countplot(x='label', data=df)
    b.set_xlabel("label")
    b.set_ylabel("Distribution")

    return
