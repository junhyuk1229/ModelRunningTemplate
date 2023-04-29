from .model.model_mlp import MultiLayerPerceptronClass
from .model.model_mf import MatrixFactorization


def create_model(settings: dict):
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
    elif settings["run_model"]["name"].lower() == "mf":
        model = MatrixFactorization(settings)

    print("Created Model!")
    print()

    return model
