import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel


class LongShortTermMemoryAttention(nn.Module):
    def __init__(self, data, settings):
        super().__init__()

        self.device = settings["device"]

        self.hidden_dim = settings["lstm_attn"]["hidden_dim"]
        self.input_embed_dim = settings["lstm_attn"]["input_dim"]
        self.lstm_input_dim = settings["lstm_attn"]["lstm_input_dim"]
        self.n_layers = settings["lstm_attn"]["n_layers"]
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

        self.n_heads = settings["lstm_attn"]["n_heads"]
        self.drop_out = settings["lstm_attn"]["drop_out"]

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )

        self.attn = BertEncoder(self.config).to(self.device)

    def forward(self, x):
        input_size = len(x["interaction"])

        embedded_x = torch.cat(
            [self.embedding[i](x[i].int()) for i in list(self.n_input_list)], dim=2
        )

        input_x = self.input_lin(embedded_x)

        output_x, _ = self.lstm(input_x)

        output_x = output_x.contiguous().view(input_size, -1, self.hidden_dim)

        extended_attention_mask = x["mask"].unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(
            output_x, extended_attention_mask, head_mask=head_mask
        )
        sequence_output = encoded_layers[-1]

        y_hat = self.output_lin(sequence_output).view(input_size, -1)

        return y_hat
