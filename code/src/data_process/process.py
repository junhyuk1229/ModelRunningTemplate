import pandas as pd
from .process_age import age_average_fill_na


def process_mlp(data: pd.DataFrame) -> None:
    """
    Processes data for MLP training.

    Parameters:
        data(DataFrame): Merged input data.
    """

    age_average_fill_na(data)

    return


def process_data(data: dict, settings: dict) -> None:
    """
    Merges then modifies data and drops columns.

    Parameters:
        data(dict): Dictionary containing the unprocessed dataframes.
        settings(dict): Dictionary containing the settings.
    """

    print("Merging Data...")

    # Merge data
    merged_data = (
        data["train_ratings"]
        .merge(data["user_data"], on="user_id")
        .merge(data["book_data"], on="isbn")
    )

    print("Merged Data!")
    print()

    print("Modifing Data...")

    # Modify data
    if settings["model"]["name"].lower() == "mlp":
        process_mlp(merged_data)

    # Drop unwanted columns
    merged_data = merged_data[settings["choose_columns"]]

    print("Modified Data!")
    print()

    # Save processed data to data dictionary
    data["processed_data"] = merged_data

    return
