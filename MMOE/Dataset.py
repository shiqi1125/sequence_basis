import torch
from torch.utils.data import Dataset
import numpy as np
from pyfaidx import Fasta
import math
import matplotlib.pyplot as plt
from Utils import *

class GenomicDataset(Dataset):
    def __init__(self, file_config, train_config):
        self.window_size = train_config['window_size']
        self.token_len = train_config['token_len']
        self.token_cnt = self.window_size // self.token_len
        self.stride = train_config['stride']
        self.holdout_chr = train_config['holdout_chr']

        self.binned_window_size = self.window_size // self.token_len
        self.binned_token_cnt = self.binned_window_size
        self.binned_stride = self.stride // self.token_len

        self.fasta_path = file_config['dna']
        self.atac_path = file_config['atac']
        self.rna_path = file_config['rna']

        # sampler
        self.train = []
        self.valid = []
        
        # create train data
        self.fasta = Fasta(self.fasta_path, sequence_always_upper=True)

        # build global token offset
        self.chrom_lengths = {chrom: len(self.fasta[chrom]) for chrom in self.fasta.keys()}
        self.chrom_offset = {}
        offset = 0
        for chrom, length in self.chrom_lengths.items():
            self.chrom_offset[chrom] = offset
            num_tokens = (length + self.token_len - 1) // self.token_len
            offset += num_tokens
        self.total_tokens = offset

        self.windows = []
        for chrom in self.fasta.keys():
            chrom_len = len(self.fasta[chrom])
            for start in range(0, chrom_len, self.stride):
                end = min(start + self.window_size, chrom_len)
                self.windows.append((chrom, start, end))
        self.atac = deserialize_pickle(self.atac_path)
        self.rna = deserialize_pickle(self.rna_path)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        chrom, start, end = self.windows[idx]
        signal_start = start // self.token_len
        # ceil division: signals sequences have already padded
        signal_end = math.ceil(end / self.token_len)
        seq = str(self.fasta[chrom][start:end])
        atac = self.atac[chrom][signal_start:signal_end]
        rna = self.rna[chrom][signal_start:signal_end]
        if len(seq) < self.window_size:
            seq = seq + 'N' * (self.window_size - len(seq))
        dna_onehot = dna_to_onehot(seq, self.window_size)                               # [128000, 4]
        dna_tokens = dna_onehot.reshape(self.token_cnt, self.token_len, 4)  # [1000, 128, 4]
        pad_bins = self.binned_window_size - len(atac)
        if pad_bins > 0:
            atac = list(atac) + [0.0] * pad_bins
            rna = list(rna) + [0.0] * pad_bins
        atac = np.array(atac).astype(np.float32).reshape(-1,1)              # [1000, 1]
        rna = np.array(rna).astype(np.float32)                              # [1000]

        # Compute global bin indices for this window
        token_offset = self.chrom_offset[chrom] + (start // self.token_len)
        chrom_token_len = (self.chrom_lengths[chrom] + self.token_len - 1) // self.token_len
        max_token_idx = self.chrom_offset[chrom] + chrom_token_len - 1
        global_token_idx = np.arange(token_offset, token_offset + self.token_cnt, dtype=np.int64)
        global_token_idx = np.clip(global_token_idx, self.chrom_offset[chrom], max_token_idx)

        return (torch.tensor(dna_tokens, dtype=torch.float32), 
                torch.tensor(atac, dtype=torch.float32),
                torch.tensor(rna, dtype=torch.float32),
                torch.tensor(global_token_idx, dtype=torch.long))
    
if __name__ == "__main__":
    file_config, train_config = {}, {}
    file_config['dna'] = '../Data/ZM/B73-5.fa'
    file_config['rna'] = '../Data/ZM/rna.pkl'
    file_config['atac'] = '../Data/ZM/atac.pkl'
    train_config['window_size'] = 128000
    train_config['token_len'] = 128
    train_config['stride'] = 128000
    train_config['holdout_chr'] = '10'

    dataset = GenomicDataset(file_config=file_config, train_config=train_config)
    # print(len(dataset.valid))
    # print(len(dataset.train))
    # print(len(dataset))
    # dna, atac, rna = dataset[5000]
    # atac = atac.cpu().numpy()
    # rna = rna.cpu().numpy()
    atac = dataset.atac
    rna = dataset.rna
    atac = atac['1'][:10000]
    rna = rna['1'][:10000]

    print(dataset.total_tokens)

    fig,axs = plt.subplots(2)
    axs[0].plot(atac)
    axs[0].set_title('atac')
    axs[1].plot(rna)
    axs[1].set_title('rna')
    plt.show()
