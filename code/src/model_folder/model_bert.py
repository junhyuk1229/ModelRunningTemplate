import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel


class BidirectionalEncoderRepresentationsfromTransformers(nn.Module):
    def __init__(self, data, settings):
        super().__init__()

        self.device = settings["device"]

        self.hidden_dim = settings["bert"]["hidden_dim"]
        self.input_embed_dim = settings["bert"]["input_dim"]
        self.lstm_input_dim = settings["bert"]["lstm_input_dim"]
        self.n_layers = settings["bert"]["n_layers"]
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

        self.n_heads = settings["bert"]["n_heads"]

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            max_position_embeddings=settings["bert"]["max_seq_len"],
        )

        self.encoder = BertModel(self.config).to(self.device)

    def forward(self, x):
        input_size = len(x["interaction"])

        embedded_x = torch.cat(
            [self.embedding[i](x[i].int()) for i in list(self.n_input_list)], dim=2
        )

        input_x = self.input_lin(embedded_x)

        encoded_layers = self.encoder(inputs_embeds=input_x, attention_mask=x["mask"])
        out = encoded_layers[0]
        out = out.contiguous().view(input_size, -1, self.hidden_dim)
        out = self.output_lin(out).view(input_size, -1)

        return out
