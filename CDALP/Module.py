import torch
import torch.nn as nn
import torch.nn.functional as F

# use DNABERT2 as sequence encoder to extract DNA sequence features
class SequenceEncoder():
    def __init__(self, tokenizer=None, seq_encoder=None):
        # Load DNABERT-2 tokenizer and model
        self.tokenizer = tokenizer
        self.seq_encoder = seq_encoder

# ATAC encoder to extract ATAC signal features
class ATACEncoder(nn.Module):
    def __init__(self, input_length=2000, emb_dim=768):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=7, padding=3)        # [B,128,2000]
        self.pool1 = nn.MaxPool1d(kernel_size=2)                        # [B,128,1000]
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)      # [B,256,1000]
        self.pool2 = nn.MaxPool1d(kernel_size=2)                        # [B,256,500]
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)      # [B,512,500]
        self.pool3 = nn.MaxPool1d(kernel_size=2)                        # [B,512,250]
        self.conv4 = nn.Conv1d(512, emb_dim, kernel_size=3, padding=1)  # [B,768,250]
        self.global_pool = nn.AdaptiveAvgPool1d(1)                      # [B,768,1]

    def forward(self, x):
        x = F.relu(self.conv1(x))       # [B, 1, 2000]
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.global_pool(x)         # [B,768,1]
        x = x.view(x.size(0), -1)       # flatten to [B,768]
        return x
    
# ATAC decoder block to reconstruct ATAC signals
class ATACDecoder(nn.Module):
    def __init__(self, emb_dim=768, atac_dim=768):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, atac_dim)
        )
    def forward(self, dna_emb, atac_emb):
        joint = torch.cat([dna_emb, atac_emb], dim=1)   # [B,1536]
        return self.decoder(joint)                      # [B,2000]