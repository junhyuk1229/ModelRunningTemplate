from itertools import permutations
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_sns_categorical(input_df: pd.DataFrame, categorical_columns: list) -> None:
    for col_first, col_second in permutations(categorical_columns, 2):
        _, ax = plt.subplots(1, 1, figsize=(20, 10))

        sns.countplot(x=col_first, data=input_df, hue=col_second, ax=ax)
        plt.show()

    return


def plot_sns_mixed(
    input_df: pd.DataFrame, categorical_columns: list, noncategorical_columns: list
):
    for col_first in categorical_columns:
        for col_second in noncategorical_columns:
            sns.boxplot(x=col_first, y=col_second, data=input_df)
        plt.show()

    return
