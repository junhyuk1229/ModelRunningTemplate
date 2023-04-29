import pandas as pd
from .data_modify import age_average_fill_na, average_fill_na


def index_data(data: dict, settings: dict) -> None:
    """
    Indexes selected columns from setting.

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes.
        settings(dict): Dictionary containing the settings.
    """
    # Loop for each indexing columns
    for index_column in settings["index_columns"]:
        # Save unique values of the column
        unique_elements = set()
        unique_elements.update(data["train_ratings"][index_column].unique().tolist())
        unique_elements.update(data["test_ratings"][index_column].unique().tolist())
        unique_elements.update(
            data["sample_submission"][index_column].unique().tolist()
        )

        # Create dictionary to change data to index
        temp_idx = {v: i for i, v in enumerate(unique_elements)}

        # Map the dictionary to values
        data["train_ratings"][index_column] = data["train_ratings"][index_column].map(
            temp_idx
        )
        data["test_ratings"][index_column] = data["test_ratings"][index_column].map(
            temp_idx
        )
        data["sample_submission"][index_column] = data["sample_submission"][
            index_column
        ].map(temp_idx)

    return


def process_mlp(data: dict) -> None:
    """
    Processes data for MLP training.

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes.
    """

    average_fill_na(data["user_data"], "age")

    return


def process_data(data: dict, settings: dict) -> None:
    """
    Merges then modifies data and drops columns.

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes.
        settings(dict): Dictionary containing the settings.
    """

    # Modify data
    print("Modifing Data...")

    if settings["run_model"]["name"].lower() == "mlp":
        process_mlp(data)

    print("Modified Data!")
    print()

    # Merge data
    print("Merging Data...")

    data["train_ratings"] = (
        data["train_ratings"]
        .merge(data["user_data"], on="user_id")
        .merge(data["book_data"], on="isbn")
    )
    data["test_ratings"] = (
        data["test_ratings"]
        .merge(data["user_data"], on="user_id")
        .merge(data["book_data"], on="isbn")
    )
    data["sample_submission"] = (
        data["sample_submission"]
        .merge(data["user_data"], on="user_id")
        .merge(data["book_data"], on="isbn")
    )

    print("Merging Data!")
    print()

    # Drop unwanted columns
    print("Dropping Columns...")

    if settings["run_model"]["name"].lower() == "mf":
        data["train_ratings"] = data["train_ratings"][["user_id", "isbn", "rating"]]
        data["test_ratings"] = data["test_ratings"][["user_id", "isbn", "rating"]]
        data["sample_submission"] = data["sample_submission"][["user_id", "isbn", "rating"]]
    else:
        data["train_ratings"] = data["train_ratings"][settings["choose_columns"]]
        data["test_ratings"] = data["test_ratings"][settings["choose_columns"]]
        data["sample_submission"] = data["sample_submission"][settings["choose_columns"]]

    # Drop predicting column for test
    data["test_ratings"] = data["test_ratings"].drop(
        columns=[settings["predict_column"]]
    )

    if settings["run_model"]["name"].lower() == "mlp":
        settings["column_num"] = len(data["test_ratings"].columns)
    elif settings["run_model"]["name"].lower() == "mf":
        settings["user_num"] = len(data["user_data"][["user_id"]])
        settings["book_num"] = len(data["book_data"][["isbn"]])

    print("Dropped Columns!")
    print()

    # Index columns
    print("Indexing columns...")

    index_data(data, settings)

    print("Indexed Columns!")
    print()

    return
