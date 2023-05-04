import pandas as pd


def average_fill_na(data: pd.DataFrame, column_name: str) -> None:
    """
    Replaces na values in the column with the average of all.

    Created By:
        Jun Hyuk Kim

    Parameters:
        data(dataframe): Merged(user, book, rating) dataframe.
        column_name(str): Name of the column to be filled.
    """

    # Fill NaN ages in total average
    data[column_name] = data[column_name].fillna(data[column_name].mean())

    return
