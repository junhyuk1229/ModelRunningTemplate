from .model.model_mlp import MultiLayerPerceptronClass


def create_model(settings):
    if settings["model"]["name"].upper() == "MLP":
        model = MultiLayerPerceptronClass()

    return model
