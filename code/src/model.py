from .model_folder.model_mlp import MultiLayerPerceptronClass
from .model_folder.model_lstm import LongShortTermMemory


def create_model(data: dict, settings: dict):
    """
    Creates model using settings.

    Parameters:
        settings(dict): Dictionary containing the settings.

    Returns:
        model(nn.Module): Model based on settings.
    """
    print("Creating Model...")

    if settings["run_model"]["name"].lower() == "mlp":
        model = MultiLayerPerceptronClass(settings, input_dim=settings["column_num"])
    if settings["run_model"]["name"].lower() == "lstm":
        model = LongShortTermMemory(data, settings)

    print("Created Model!")
    print()

    return model
