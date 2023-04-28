import pandas as pd


def age_average_fill_na(data: pd.DataFrame) -> None:
    """
    Replaces na values in the age column with the average age of all users.

    Created By:
        Jun Hyuk Kim

    Parameters:
        data(dataframe): Merged(user, book, rating) dataframe.
    """

    # Fill NaN ages in total average
    data["age"] = data["age"].fillna(data["age"].mean())

    return
