import logging
from sklearn.neural_network import MLPRegressor, MLPClassifier
import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """
    Multilayer Perceptron (MLP) Class

    Creates multiple linear layers seperated by a activation function.
    Output will always be a single value.
    If no layer_dim is given creates a generic dense layer with a single linear layer.
    """

    def __init__(self, input_dim, layer_dim, output_dim):
        super(MultiLayerPerceptron, self).__init__()

        # Input dimension
        self.input_dim = input_dim
        # Output dimension
        self.output_dim = output_dim

        # Layer dimensions
        self.layer_dim = layer_dim
        # Number of layers
        self.layer_num = len(self.layer_dim)

        # Multiple linear layers
        self.lin_list = nn.Sequential(
            nn.Linear(self.input_dim, self.layer_dim[0]), nn.ReLU()
        )

        for i in range(self.layer_num - 1):
            self.lin_list.append(nn.Linear(self.layer_dim[i], self.layer_dim[i + 1]))
            self.lin_list.append(nn.ReLU())

        self.lin_list.append(nn.Linear(self.layer_dim[-1], self.output_dim))

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


def get_mlp_classifier(**kwargs):
    return MLPClassifier(kwargs)


def get_mlp_regressor(**kwargs):
    return MLPRegressor(kwargs)
