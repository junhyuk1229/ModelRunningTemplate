import torch
import torch.nn as nn


class LongShortTermMemory(nn.Module):
    def __init__(self, data, settings):
        super().__init__()
        self.hidden_dim = settings["lstm"]["hidden_dim"]
        self.input_embed_dim = settings["lstm"]["input_dim"]
        self.lstm_input_dim = settings["lstm"]["lstm_input_dim"]
        self.n_layers = settings["lstm"]["n_layers"]
        self.n_input_list = {i: c for i, c in data["idx"].items()}

        # embedding layers
        self.embedding = dict()
        self.embedding["interaction"] = nn.Embedding(3, self.input_embed_dim)
        for i in ["testId", "assessmentItemID", "KnowledgeTag"]:
            v = self.n_input_list[i]
            self.embedding[i] = nn.Embedding(v + 1, self.input_embed_dim)

        self.input_lin = nn.Linear(
            len(self.embedding) * self.input_embed_dim, self.lstm_input_dim
        )
        self.output_lin = nn.Linear(self.hidden_dim, 1)

        self.lstm = nn.LSTM(
            self.lstm_input_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

    def forward(self, x):
        batch_size = len(x["interaction"])

        embedded_x = None
        # 임시 #
        # Embedding
        # for i, v in x.items():
        for i in ["interaction", "testId", "assessmentItemID", "KnowledgeTag"]:
            v = x[i]
            # 임시 #
            if i in ["answerCode", "mask"]:
                continue
            if embedded_x is None:
                embedded_x = self.embedding[i](v.int())
            else:
                embedded_x = torch.cat([embedded_x, self.embedding[i](v.int())], axis=2)

        input_x = self.input_lin(embedded_x)

        output_x, _ = self.lstm(input_x)

        output_x = output_x.contiguous().view(batch_size, -1, self.hidden_dim)
        y_hat = self.output_lin(output_x)

        return y_hat.view(batch_size, -1)
