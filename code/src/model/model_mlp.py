import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptronClass(nn.Module):
    """
    Multilayer Perceptron (MLP) Class
    """

    def __init__(self, settings, input_dim):
        super(MultiLayerPerceptronClass, self).__init__()

        self.layer_num = settings["mlp"]["layer_num"]
        self.layer_dim = settings["mlp"]["layer_dim"]

        self.lin = nn.Sequential(nn.Linear(input_dim, self.layer_dim[0]), nn.ReLU())

        for i in range(self.layer_num - 2):
            self.lin.append(nn.Linear(self.layer_dim[i], self.layer_dim[i + 1]))
            self.lin.append(nn.ReLU())

        self.lin.append(nn.Linear(self.layer_dim[-1], 1))

    def forward(self, x: torch.Tensor):
        y_hat = self.lin(x)

        return y_hat
