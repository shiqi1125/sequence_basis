import torch
from torch.utils.data import Dataset, DataLoader

import random
import numpy as np

class MockGenomicsDataset(Dataset):
    def __init__(self, tokenizer=None, num_samples=1000, seq_length=2000):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.dna_seqs = []
        self.atac_signals = []
        self.labels = []
        bases = ['A', 'C', 'G', 'T']
        for _ in range(num_samples):
            seq = ''.join(random.choices(bases, k=seq_length))
            self.dna_seqs.append(seq)
            self.atac_signals.append(np.random.randn(seq_length).astype(np.float32))
            self.labels.append(random.randint(0, 2))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        dna = self.dna_seqs[idx]
        atac = torch.from_numpy(self.atac_signals[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        # Tokenize DNA
        encoding = self.tokenizer(dna, return_tensors='pt', truncation=True, max_length=self.seq_length)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask, atac, label

if __name__ == '__main__':
    dataset = MockGenomicsDataset(num_samples=1, seq_length=10)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(dataset.dna_seqs)
    print(dataset.atac_signals)
    print(dataset.labels)