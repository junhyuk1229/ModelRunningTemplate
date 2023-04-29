import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptronClass(nn.Module):
    """
    Multilayer Perceptron (MLP) Class
    """

    def __init__(self, xdim: int = 784, hdim: int = 256, ydim: int = 10):
        super(MultiLayerPerceptronClass, self).__init__()
        self.xdim = xdim
        self.hdim = hdim
        self.ydim = ydim
        self.lin_1 = nn.Linear(self.xdim, self.hdim)
        self.lin_2 = nn.Linear(self.hdim, self.ydim)
        self.init_param()

    def name(self) -> str:
        return "MLP"

    def init_param(self):
        nn.init.kaiming_normal_(self.lin_1.weight)
        nn.init.zeros_(self.lin_1.bias)
        nn.init.kaiming_normal_(self.lin_2.weight)
        nn.init.zeros_(self.lin_2.bias)

    def forward(self, x: torch.Tensor):
        net = x
        net = self.lin_1(net)
        net = F.relu(net)
        net = self.lin_2(net)
        return net
