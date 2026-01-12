import torch
import torch.nn as nn
import math

# ==============================================================================
# TRANSFORMER MODEL ARCHITECTURE (Must match training)
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe shape: [max_len, 1, d_model] -> batch_first=True in logic below
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, Seq_Len, D_Model]
        # Add positional encoding up to the sequence length of x
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CryptoTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        output_dim,
        dropout=0.1,
        max_len=500,
    ):
        super(CryptoTransformer, self).__init__()

        # 1. Input Projection: Map features to d_model size
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # 3. Transformer Encoder
        # batch_first=True means input is (Batch, Seq, Feature)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 4. Output Head
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # src: (Batch, Seq_Len, Features)

        # Embed and add position info
        src = self.input_embedding(src)
        src = self.pos_encoder(src)

        # Pass through Transformer
        # output: (Batch, Seq_Len, d_model)
        output = self.transformer_encoder(src)

        # We take the LAST time step for prediction
        # output[:, -1, :] shape: (Batch, d_model)
        return self.decoder(output[:, -1, :])
