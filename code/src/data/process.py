import pandas as pd
from .data_modify import age_average_fill_na, average_fill_na
from sklearn.preprocessing import LabelEncoder


def index_data(data: dict, settings: dict) -> None:
    """
    Processes data for MLP training.

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes.
        settings(dict): Dictionary containing the settings.
    """
    idx = dict()

    # Loop for each indexing columns
    for index_column in settings["index_columns"]:
        # Save unique values of the column
        le = LabelEncoder()
        unique_value = data["train"][index_column].unique().tolist() + ["unknown"]
        le.fit(unique_value)

        data["test"][index_column] = data["test"][index_column].apply(
            lambda x: x if str(x) in le.classes_ else "unknown"
        )

        # Map the dictionary to values
        data["train"][index_column] = le.transform(data["train"][index_column])
        data["test"][index_column] = le.transform(data["test"][index_column])

        idx[index_column] = len(unique_value)

    return idx


def process_mlp(data: dict) -> None:
    """
    Processes data for MLP training.

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes.
    """

    average_fill_na(data["user_data"], "age")

    return


def process_lstm(data: dict) -> None:
    return


def process_data(data: dict, settings: dict) -> None:
    """
    Merges then modifies data and drops columns.

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes.
        settings(dict): Dictionary containing the settings.
    """
    for df_name in list(data):
        data[df_name] = data[df_name].sort_values(by=["userID", "Timestamp"], axis=0)

    # Modify data
    print("Modifing Data...")

    if settings["run_model"]["name"].lower() == "mlp":
        process_mlp(data)
    if settings["run_model"]["name"].lower() == "lstm":
        process_lstm(data)

    print("Modified Data!")
    print()

    # Drop unwanted columns
    print("Dropping Columns...")

    data["train"] = data["train"][settings["choose_columns"]]
    data["test"] = data["test"][settings["choose_columns"]]

    print("Dropped Columns!")
    print()

    # Index columns
    print("Indexing Columns...")

    data["idx"] = index_data(data, settings)

    print("Indexed Columns!")
    print()

    return
