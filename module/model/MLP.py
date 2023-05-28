import logging
from sklearn.neural_network import MLPRegressor, MLPClassifier
import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """
    Multilayer Perceptron (MLP) Class

    Creates multiple linear layers seperated by a activation function.
    Output will always be a single value.
    If no hidden_dims is given creates a generic dense layer with a single linear layer.
    """

    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        # Input dimension
        self.input_dim = input_dim
        # Output dimension
        self.output_dim = output_dim

        # Layer dimensions
        self.hidden_dims = hidden_dims
        # Number of layers
        self.layer_num = len(self.hidden_dims)

        # Multiple linear layers
        self.lin_list = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dims[0]), nn.ReLU()
        )

        for i in range(self.layer_num - 1):
            self.lin_list.append(
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1])
            )
            self.lin_list.append(nn.ReLU())

        self.lin_list.append(nn.Linear(self.hidden_dims[-1], self.output_dim))

        self.lin_list.apply(self.init_param)

        logging.debug(
            f"Created MLP with layers: {self.input_dim}, {', '.join(map(str, self.hidden_dims))}, {self.output_dim}"
        )

    def init_param(self, lin_layer):
        if type(lin_layer) == nn.Linear:
            nn.init.kaiming_normal_(lin_layer.weight)
            nn.init.zeros_(lin_layer.bias)

    def forward(self, x: torch.Tensor):
        logging.debug(f"MLP input shape: {x.shape}")

        y_hat = self.lin_list(x)

        logging.debug(f"MLP output shape: {y_hat.shape}")

        return y_hat


def get_mlp_classifier_sklearn(**kwargs):
    """
    Gets MLP Classifier from sklearn.

    Parameters:
        **kwargs(dict): The arguments passed to the MLP Classifier.

    Returns:
        model(MLPClassifier): MLP Classifier from .
    """

    model = MLPClassifier(**kwargs)

    logging.debug(f"Created MLP Classifier model from sklearn")

    return model


def get_mlp_regressor_sklearn(**kwargs):
    """
    Gets MLP Regressor from sklearn.

    Parameters:
        **kwargs(dict): The arguments passed to the MLP Regressor.

    Returns:
        model(MLPRegressor): The optimizer with the parameters.
    """

    model = MLPRegressor(**kwargs)

    logging.debug(f"Created MLP Regressor model from sklearn")

    return model
