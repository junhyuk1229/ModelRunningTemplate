import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import MSELoss


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y) + self.eps)
        return loss


def train_model(dataloader, settings, model):
    loss_fn = RMSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(settings["train"]["epoch"]):
        model.train()
        total_loss = 0
        batch_count = 0

        for idx, data in enumerate(dataloader["train_dataloader"]):
            x, y = data
            y_hat = model(x)
            loss = loss_fn(y, torch.squeeze(y_hat))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
            batch_count += 1

        print(total_loss / batch_count)

    return