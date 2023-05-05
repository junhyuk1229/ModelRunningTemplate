import pandas as pd


def create_feature_big_tag(data: pd.DataFrame) -> None:
    data["train"]["big_tag"] = data["train"]["assessmentItemID"].apply(lambda x: x[2])
    data["test"]["big_tag"] = data["test"]["assessmentItemID"].apply(lambda x: x[2])

    return
