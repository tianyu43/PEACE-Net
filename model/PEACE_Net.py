import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    '''
     Positional Encoding, PE
    '''
    def __init__(self, d_model, pe_tau, max_seq_len):
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
        # pe: (max_seq_len, d_model) => (max_seq_len, 1, d_model)
        pe = pe.unsqueeze(0).permute((1, 0, 2))
        self.register_buffer("pe", pe)

    def forward(self, x):
        # (sql_len, batch, d_model) + (sql_len, 1, d_model)
        # => (sql_len, batch, d_model)
        return x + self.pe[:x.shape[0], :, :]



class Transformer_encoder(nn.Module):
    '''
     A transformer encoder with different pooling mechanism.
    '''
    def __init__(
        self, seed, pe_tau, input_feature_size, d_model, nhead,
        dim_feedforward, dropout, num_layers, seq_len,
        pooling="cls"
    ):
        super().__init__()
        self._set_reproducible(seed)
        
        self.fc1 = nn.Sequential(
            nn.Linear(input_feature_size, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        
        self.pos_encoding = PositionalEncoding(
            d_model=d_model, pe_tau=pe_tau, max_seq_len=seq_len
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        
        encoder_norm = nn.LayerNorm(d_model)
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, encoder_norm
        )
        
        # linear attention layer
        self.linear_attention = nn.Linear(
            in_features=d_model, out_features=1
        )
        
        # CLS token 
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model)
        )
        
        self.pooling = pooling
        if self.pooling not in ["cls", "linear", "mean"]:
            raise ValueError(f"Unknown pooling type: {self.pooling}")
    
    def _set_reproducible(self, seed, cudnn=False):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    def forward(self, x):
        # x: (B, T, F) → (T, B, F)
        x = x.permute(1, 0, 2)
        fc1_out = self.fc1(x)  # (T, B, D)

        if self.pooling == "cls":
            B = fc1_out.size(1)
            cls_tokens = self.cls_token.expand(-1, B, -1)   # (1, B, D)
            encoder_in = torch.cat([cls_tokens, fc1_out], dim=0)
            encoder_in = self.pos_encoding(encoder_in)
            encoder_out = self.encoder(encoder_in)          # (T+1, B, D)
            return encoder_out[0]                           # (B, D)

        elif self.pooling == "linear":
            encoder_in = self.pos_encoding(fc1_out)
            encoder_out = self.encoder(encoder_in)          # (T, B, D)
            encoder_out = encoder_out.permute(1, 0, 2)      # (B, T, D)
            attn_weights = F.softmax(F.relu(self.linear_attention(encoder_out)), dim=1)  # (B, T, 1)
            pooled = attn_weights.permute(0, 2, 1).bmm(encoder_out)                      # (B, 1, D)
            return pooled.squeeze(1)                         # (B, D)

        elif self.pooling == "mean":
            encoder_in = self.pos_encoding(fc1_out)
            encoder_out = self.encoder(encoder_in)          # (T, B, D)
            encoder_out = encoder_out.permute(1, 0, 2)      # (B, T, D)
            return encoder_out.mean(dim=1)                  # (B, D)





class PEACE_Net(nn.Module):
    def __init__(
        self, pe_tau, SEED, 
        input_feature_size, seq_len,
        d_model, dim_feedforward, projection_dim,
        nhead, num_layers, n_classes,
        dropout,
        pooling="cls"
        ):
        super().__init__()
        self._set_reproducible(SEED)
        
        self.encoder = Transformer_encoder(
            seed = SEED, pe_tau = pe_tau, 
            input_feature_size = input_feature_size, seq_len = seq_len,
            d_model = d_model, 
            dim_feedforward = dim_feedforward, 
            nhead = nhead, 
            num_layers = num_layers,
            dropout = dropout,
            pooling = pooling
        )
        
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes)
        )
    
    def _set_reproducible(self, seed, cudnn=False):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def forward(self, x, return_class_only=False):  # x: (B, T, F)
        # (B, D)
        encoder_feature = self.encoder(x)                    
        # (B, 2)
        class_out = self.classification_head(encoder_feature)   
        if return_class_only:
            return class_out
        else:
            # (B, projection_dim)
            proj_feature = self.projection_head(encoder_feature)               
            contrastive_out = F.normalize(proj_feature, dim=1, p=2)    
            return contrastive_out, class_out, encoder_feature, proj_feature


    def predict(self, x):
        # (B, D)
        encoder_feature = self.encoder(x)                    
        # (B, 2)
        class_out = self.classification_head(encoder_feature)
        return class_out, encoder_feature.squeeze()
