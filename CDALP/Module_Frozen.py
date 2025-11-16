import torch
import torch.nn as nn
import torch.nn.functional as F

# use DNABERT2 as sequence encoder to extract DNA sequence features
class SequenceEncoder(nn.Module):
    def __init__(self, tokenizer=None, seq_encoder=None, output_dim=128):
        super().__init__()
        # Load DNABERT-2 tokenizer and model
        self.tokenizer = tokenizer
        self.seq_encoder = seq_encoder
        # Freeze DNABERT2 weights
        for param in self.seq_encoder.parameters():
            param.requires_grad = False
        # Add a learnable linear projection
        self.projection = nn.Linear(self.seq_encoder.config.hidden_size, output_dim)

    def forward(self, tokens):
        outputs = self.seq_encoder(tokens)
        hidden = outputs[0]                         # [B, 2000, 768]
        pooled = hidden.mean(dim=1)                 # [B, 768]
        projected = self.projection(pooled)         # [B, 128]
        return projected

# ATAC encoder to extract ATAC signal features
class ATACEncoder(nn.Module):
    def __init__(self, input_length=2000, emb_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)        # [B,16,2000]
        self.pool1 = nn.MaxPool1d(kernel_size=2)                       # [B,16,1000]
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)       # [B,32,1000]
        self.pool2 = nn.MaxPool1d(kernel_size=2)                       # [B,32,500]
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)       # [B,64,500]
        self.global_pool = nn.AdaptiveAvgPool1d(1)                     # [B,64,1]
        self.fc = nn.Linear(64, emb_dim)                               # [B,128]

    def forward(self, x):
        x = F.relu(self.conv1(x))       # [B, 1, 2000]
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)         # [B,64,1]
        x = x.view(x.size(0), -1)       # [B,64]
        x = self.fc(x)                  # [B,128]
        return x

class ATACEncoder2(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, embed_dim)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CrossAttentionDecoder(nn.Module):
    def __init__(self, emb_dim=128, hidden_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.final_proj = nn.Linear(emb_dim, 2000)
    def forward(self, atac_emb, dna_emb):
        atac_emb = atac_emb.unsqueeze(0)
        dna_emb = dna_emb.unsqueeze(0)
        dec_out = self.decoder(atac_emb, dna_emb).squeeze(0)
        return self.final_proj(dec_out)

class CrossAttentionDecoder2(nn.Module):
    def __init__(self, emb_dim=768, hidden_dim=1024, num_heads=6, num_layers=2):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(emb_dim, 5000)
    def forward(self, atac_emb, dna_emb):
        atac_emb = atac_emb.unsqueeze(0)
        dna_emb = dna_emb.unsqueeze(0)
        dec_out = self.decoder(atac_emb, dna_emb).squeeze(0)
        dec_out = self.fc(dec_out)
        dec_out = F.softplus(dec_out)
        return dec_out

class NBDispersion(nn.Module):
    def __init__(self, init_r=0.1):
        super().__init__()
        # we need to constrain dispersion positive
        # inverse softplus
        self.rho = nn.Parameter(torch.log(torch.exp(torch.tensor(init_r)) - 1.0))

    @property
    def r(self):
        return F.softplus(self.rho)

class DNABaseline(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(emb_dim, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, 5000))
        
    def forward(self, dna_emb):
        return self.fc(dna_emb)