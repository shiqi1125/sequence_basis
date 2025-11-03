import torch
import json
import torch.nn.functional as F
import random
from torch.utils.data import Dataset

class Tokenize():
    def __init__(self, k, seq, augmentation_prob=-1.0):
        self.base_to_idx = {'A':0, 'C':1, 'G':2, 'T':3, 'N':0}
        self.vocab_size = 4 ** k
        self.seq = seq
        self.seq_len = len(seq)
        self.tokens = []
        data = None
        if k == 6:
            with open('6-mers.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
        k_mers = self.construct_k_mers(k, seq)
        for elem in k_mers:
            self.tokens.append(data.get(elem, 0))
        #self.token_ids = torch.tensor(self.tokens, dtype=torch.long)

        # data augmentation
        if augmentation_prob > 0.0:
            self.aug_tokens = []
            self.augmented_seq = self.random_reverse_complement(self.seq, augmentation_prob)
            aug_k_mers = self.construct_k_mers(k, self.augmented_seq)
            for elem in aug_k_mers:
                self.aug_tokens.append(data.get(elem, 0))
            self.aug_token_ids = torch.tensor(self.aug_tokens, dtype=torch.long)

    def reverse_complement(self, seq):
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
        comp_seq = "".join(complement[base] for base in seq)
        return comp_seq[::-1]
    
    def random_reverse_complement(self, seq, p=0.5):
        if random.random() < p:
            return self.reverse_complement(seq)
        return seq

    def construct_k_mers(self, k, seq):
        result = []
        seq_len = len(seq)
        if seq_len >= k:
            for i in range(seq_len - k + 1):
                result.append(''.join(seq[i:i+k]).upper())
        else:
            result.append(''.join(seq).ljust(k, 'N').upper())
        return result

class GenomicDataset(Dataset):
    def __init__(self, token_ids, tgt_values, window_size, overlap):
        self.token_ids = token_ids
        self.tgt_values = tgt_values
        self.window_size = window_size
        self.stride = window_size - overlap
        self.seq_len = len(token_ids)
        self.starts = list(range(0, self.seq_len, self.stride))
        if self.starts[-1] + window_size < self.seq_len:
            self.starts.append(self.seq_len - window_size)
        self.starts = sorted(set(self.starts))

    def __len__(self):
        return len(self.starts)
    
    def __getitem__(self, idx):
        start = self.starts[idx]
        end = min(start + self.window_size, self.seq_len)
        seq_window = self.token_ids[start:end]
        tgt_window = self.tgt_values[start:end]
        if len(seq_window) < self.window_size:
            pad_len = self.window_size - len(seq_window)
            seq_window = F.pad(seq_window, (0, pad_len), value=0)
            tgt_window = F.pad(tgt_window, (0, pad_len), value=0.0)
        return seq_window, tgt_window


if __name__ == '__main__':
    window_size = 16
    overlap = 4
    stride = window_size - overlap
    from SignalLoader import SignalLoader
    data = SignalLoader('resources/tair10-3.fa', 'resources/peat.sorted.bw')
    seq = data.get_seq('Chr1', 0, 10)
    tgt_signal = data.get_signal('Chr1', 0, 10)
    tokens = Tokenize(6,seq)
    token_ids = tokens.token_ids
    print(token_ids)
