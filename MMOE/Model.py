import torch
import torch.nn as nn
import math

class GlobalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_tokens):
        super(GlobalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_tokens, d_model)
        position = torch.arange(0, max_tokens).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x, token_idx):
        # print("bin_idx min:", token_idx.min().item(), "max:", token_idx.max().item(), "max_bins:", self.pe.shape[0])
        # assert (token_idx >= 0).all() and (token_idx < self.pe.shape[0]).all(), "bin_idx out of bounds!"
        pos_emb = self.pe[token_idx]
        return x + pos_emb

class DNAEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, nlayers=2, max_tokens=100000):
        super().__init__()
        self.cnn = nn.Conv1d(4, d_model, kernel_size=9, padding=4)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model*2, batch_first=True), nlayers)
        self.global_pos_enc = GlobalPositionalEncoding(d_model, max_tokens)
    def forward(self, x, token_idx):                                   # [batch, token_cnt, token_len, 4]
        b, t_cnt, t_len, c = x.shape
        x = x.view(b*t_cnt, t_len, c).permute(0, 2, 1)      # [batch * token_cnt, 4, d_model]
        # print("CNN input shape:", x.shape)
        # assert x.shape[1] == 4 and x.shape[2] == 128, f"Bad input shape: {x.shape}"
        x = self.cnn(x).mean(dim=2)                         # [batch * token_cnt, d_model]
        x = x.view(b, t_cnt, -1)                            # [batch, token_cnt, d_model]
        x = self.global_pos_enc(x, token_idx)
        x = self.transformer(x)
        return x                                            # [batch, token_cnt, d_model]

class ATACEncoder(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.cnn = nn.Conv1d(1, d_model, kernel_size=5, padding=2)
    def forward(self, x):                    # [batch, token_cnt, 1]
        b, t_cnt, c = x.shape
        x = x.view(b*t_cnt, c, 1)            # [batch * token_cnt, 1, 1]
        x = self.cnn(x).squeeze(2)           # [batch * token_cnt, d_model]
        x = x.view(b, t_cnt, -1)             # [batch， token_cnt, d_model]
        return x

class ATACRouter(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.cnn = nn.Conv1d(1, 16, kernel_size=9, padding=4)
        self.fc = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):                   # [batch, token_cnt, 1]
        b, t_cnt, c = x.shape
        x = x.view(b*t_cnt, c, 1)           # [batch * token_cnt, 1, 1]
        x = self.cnn(x).squeeze(2)          # [batch * token_cnt, 16]
        logits = self.fc(x)                 # [batch * token_cnt, 2]
        route = torch.argmax(logits, dim=1).view(b, t_cnt)   # [batch, token_cnt]
        return route

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
    def forward(self, dna, atac):
        attn_out, _ = self.cross_attn(dna, atac, atac)
        return attn_out


class GenomicsModel(nn.Module):
    def __init__(self, d_model=128, nhead=4, nlayers=2, max_tokens=100000):
        super().__init__()
        self.dna_encoder = DNAEncoder(d_model, nhead, nlayers, max_tokens)
        self.atac_encoder = ATACEncoder(d_model)
        self.cross_attn_block = CrossAttentionBlock(d_model, nhead)
        self.atac_router = ATACRouter(d_model)
        self.output = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, dna, atac, token_idx):
        # Encode
        dna_encoded = self.dna_encoder(dna, token_idx)         # [batch, token_cnt, d_model]
        atac_encoded = self.atac_encoder(atac)                 # [batch, token_cnt, d_model]
        cross_encoded = self.cross_attn_block(dna_encoded, atac_encoded) # [batch, token_cnt, d_model]
        routes = self.atac_router(atac)             # [batch, token_cnt], 0: DNA, 1: Cross
        # Routing
        out = dna_encoded.clone()
        mask = (routes == 1)  # [batch, token_cnt], bool
        out = torch.where(mask.unsqueeze(-1), cross_encoded, dna_encoded)
        out = self.output(out).squeeze(-1)          # [batch, token_cnt]
        return out
