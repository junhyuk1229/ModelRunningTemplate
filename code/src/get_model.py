from .model.model_mlp import MultiLayerPerceptronClass


def create_model(settings):
    if settings["model"]["name"].upper() == "MLP":
        model = MultiLayerPerceptronClass(xdim=4, hdim=2, ydim=1)

    return model
