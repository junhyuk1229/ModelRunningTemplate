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

    if settings["model"]["name"].upper() == "MLP":
        model = MultiLayerPerceptronClass(xdim=4, hdim=2, ydim=1)

    print("Created Model!")
    print()

    return model
