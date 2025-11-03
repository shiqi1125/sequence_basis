import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

class GenomicTransformModel(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 d_model=128, 
                 nhead=8, 
                 num_layers=4, 
                 dim_feedforward=256, 
                 dropout=0.1, 
                 window_size=1024,
                 local_window=128,
                 gradient_checkpoint=False):
        super().__init__()
        self.d_model = d_model
        self.gradient_checkpoint = gradient_checkpoint
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_encoding", self._create_positional_encoding(max_len=window_size, d_model=d_model))
        self.conv1d = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=7, padding=3)
        self.register_buffer("attention_mask", self._create_local_attention_mask(window_size, local_window))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_linear = nn.Linear(d_model, 1)

    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-torch.log(torch.tensor(10000.0))/d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def _create_local_attention_mask(self, seq_len, local_window):
        mask = torch.full((seq_len, seq_len), float('-inf'))
        for i in range(seq_len):
            start = max(0, i - local_window)
            end = min(seq_len, i + local_window + 1)
            mask[i, start:end] = 0.0
        return mask

    def forward(self, token_batch):
        batch_size, seq_len = token_batch.shape
        input = self.embedding(token_batch) + self.pos_encoding[:, :seq_len, :]
        input = nn.functional.dropout(input, p=0.1, training=self.training)

        # # pass through 1-D conv layer
        # # (batch, d_model, seq_len)
        # input = input.transpose(1, 2)
        # input = self.conv1d(input)
        # # (batch_size, seq_len, d_model)
        # input = input.transpose(1, 2)

        # (seq_len, batch_size, d_model)
        input = input.transpose(0,1)
        input = self.transformer_encoder(input, mask=self.attention_mask)
        input = self.norm(input)
        # (batch_size, seq_len, d_model)
        input = input.transpose(0,1)
        # (batch_size, seq_len, 1)
        output = self.output_linear(input)
        output = F.softplus(output)
        # (batch_size, seq_len)
        return output.squeeze(-1)


