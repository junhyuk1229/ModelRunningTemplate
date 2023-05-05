import torch
import torch.nn as nn


class LongShortTermMemory(nn.Module):
    def __init__(self, data, settings):
        super().__init__()

        self.device = settings["device"]

        self.hidden_dim = settings["lstm"]["hidden_dim"]
        self.input_embed_dim = settings["lstm"]["input_dim"]
        self.lstm_input_dim = settings["lstm"]["lstm_input_dim"]
        self.n_layers = settings["lstm"]["n_layers"]
        self.n_input_list = data["idx"]

        # embedding layers
        self.embedding = dict()
        self.embedding["interaction"] = nn.Embedding(3, self.input_embed_dim).to(
            self.device
        )
        for i, v in self.n_input_list.items():
            self.embedding[i] = nn.Embedding(v + 1, self.input_embed_dim).to(
                self.device
            )

        self.n_input_list["interaction"] = 3

        self.input_lin = nn.Linear(
            len(self.embedding) * self.input_embed_dim, self.lstm_input_dim
        ).to(self.device)
        self.output_lin = nn.Linear(self.hidden_dim, 1).to(self.device)

        self.lstm = nn.LSTM(
            self.lstm_input_dim, self.hidden_dim, self.n_layers, batch_first=True
        ).to(self.device)

    def forward(self, x):
        input_size = len(x["interaction"])

        embedded_x = torch.cat(
            [self.embedding[i](x[i].int()) for i in list(self.n_input_list)], dim=2
        )

        input_x = self.input_lin(embedded_x)

        output_x, _ = self.lstm(input_x)

        output_x = output_x.contiguous().view(input_size, -1, self.hidden_dim)

        y_hat = self.output_lin(output_x).view(input_size, -1)

        return y_hat
