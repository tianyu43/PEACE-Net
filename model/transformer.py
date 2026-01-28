import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, pe_tau, max_seq_len=5000):
        super().__init__()

        # pe: positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).float().unsqueeze(1)
        divisor = -torch.exp(
            torch.arange(0, d_model, 2).float()
            * math.log(pe_tau) / d_model
        )
        pe[:, 0::2] = torch.sin(position * divisor)
        pe[:, 1::2] = torch.cos(position * divisor)
        # pe: (max_seq_len, d_model) => (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # (batch, seq_len, d_model) + (1, seq_len, d_model)
        return x + self.pe[:, :x.shape[1], :]


class TransformerClassifier(nn.Module):
    def __init__(
        self, pe_tau, SEED,
        input_feature_size, seq_len,
        d_model, dim_feedforward,
        nhead, num_layers,
        dropout, n_classes
    ):
        super().__init__()
        self._set_reproducible(SEED)
        
        self.fc1 = nn.Linear(
            in_features=input_feature_size, out_features=d_model
        )  # increase dimension
        self.pos_encoding = PositionalEncoding(
            d_model=d_model, pe_tau=pe_tau, max_seq_len=seq_len
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, encoder_norm
        )
        self.pool = nn.AvgPool1d(seq_len)
        self.fc2 = nn.Linear(
            in_features=d_model,
            out_features=n_classes,
        )
    
    def _set_reproducible(self, seed, cudnn=False):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    def forward(self, x):
        # x: (batch, seq_len, input_feature_size)
        # fc1_out: (batch, seq_len, d_model)
        fc1_out = self.fc1(x)
        # encoder_in: (batch, seq_len, d_model)
        encoder_in = self.pos_encoding(fc1_out)
        # encoder_out: (batch, seq_len, d_model)
        encoder_out = self.encoder(encoder_in)
        # pool_out: (batch, d_model, 1)
        pool_out = self.pool(encoder_out.transpose(1, 2))
        # outputs: (batch, num_classes)
        return self.fc2(pool_out.squeeze()), pool_out.squeeze()
