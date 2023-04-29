from .model.model_mlp import MultiLayerPerceptronClass


def create_model(settings: dict):
    """
    Creates model using settings.

    Parameters:
        settings(dict): Dictionary containing the settings.

    Returns:
        model(nn.Module): Model based on settings.
    """
    print("Creating Model...")

    if settings["run_model"]["name"].upper() == "MLP":
        model = MultiLayerPerceptronClass(settings, input_dim=settings["column_num"])

    print("Created Model!")
    print()

    return model
