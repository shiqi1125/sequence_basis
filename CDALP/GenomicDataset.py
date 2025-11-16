import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from DataProcessor import SeqEpigenPairPicker, SeqProcessor, EpigenProcessor, CreateExpr

class GenomicsDataset(Dataset):
    def __init__(self, tokenizer=None, seq_length=2000, threshold=1.0, non_peak_percentage=0.1):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.threshold = threshold
        self.dna_seqs = []
        self.atac_signals = []
        self.rna_signals = []
        self.npp = non_peak_percentage

    def generate_atac_data(self):
        # generate pairs
        fa_name = 'resources/HC04.fasta'
        bw_name = 'resources/atac_bw/L21_1_R1.trim.srt.nodup.no_HC04_CM.tn5.pval0.01.500K.bfilt.bw'
        sp = SeqProcessor(fa_name)
        ep = EpigenProcessor(bw_name)
        pp = SeqEpigenPairPicker(sp, ep, seq_len=self.seq_length, threshold=self.threshold)
        pairs = pp.generate_whole_peak_pairs()
        for pair in pairs:
            self.dna_seqs.append(pair[0])
            self.atac_signals.append(pair[1])
        self.num_samples = len(pairs)

    def all_expr(self):
        # generate pairs
        fa_name = 'resources/HC04.fasta'
        bw_name = 'resources/atac_bw/L21_1_R1.trim.srt.nodup.no_HC04_CM.tn5.pval0.01.500K.bfilt.bw'
        csv_name = 'resources/cotton_rnaseq_true.csv'
        sample_name = 'L021'
        cache_name = 'cache/' + '.'.join([sample_name, 'whole'])
        sp = SeqProcessor(fa_name)
        ep = EpigenProcessor(bw_name)
        pp = SeqEpigenPairPicker(sp, ep, seq_len=self.seq_length)
        ce = CreateExpr(sp=sp, file_name=csv_name, sample_name=sample_name, use_cache=True)
        expr = ce.make_expr()
        pairs = pp.generate_whole_expr_data(expr, self.npp, expr_threshold=1.0, use_cache=True, cache_name=cache_name)
        for pair in pairs:
            self.dna_seqs.append(pair[0])
            self.atac_signals.append(pair[1])
            self.rna_signals.append(pair[2])
        self.num_samples = len(pairs)

    def chrom_expr(self, sample_name, chrom_id):
        # generate pairs
        fa_name = 'resources/HC04.fasta'
        bw_name = 'resources/atac_bw/L21_1_R1.trim.srt.nodup.no_HC04_CM.tn5.pval0.01.500K.bfilt.bw'
        csv_name = 'resources/cotton_rnaseq_true.csv'
        cache_name = 'cache/' + '.'.join([sample_name, 'whole'])
        sp = SeqProcessor(fa_name)
        ep = EpigenProcessor(bw_name)
        pp = SeqEpigenPairPicker(sp, ep, seq_len=self.seq_length)
        ce = CreateExpr(sp=sp, file_name=csv_name, sample_name=sample_name, use_cache=True)
        expr = ce.make_expr()
        pairs = pp.gen_expr_data_by_chrom_id_cond(chrom_id, expr, -10)
        for pair in pairs:
            self.dna_seqs.append(pair[0])
            self.atac_signals.append(pair[1])
            self.rna_signals.append(pair[2])
        self.num_samples = len(pairs)

    def clear(self):
        self.dna_seqs = []
        self.atac_signals = []
        self.rna_signals = []

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        dna = self.dna_seqs[idx]
        atac = torch.from_numpy(self.atac_signals[idx])
        rna = torch.from_numpy(self.rna_signals[idx])
        # Tokenize DNA
        encoding = self.tokenizer(dna, truncation=True, max_length=self.seq_length)
        # dna_batch = {'input_ids':torch.tensor(encoding['input_ids']).squeeze(0),
        #              'attention_mask':torch.tensor(encoding['attention_mask']).squeeze(0)}
        return dna, atac, rna

class ATACDataset(Dataset):
    def __init__(self, tokenizer=None, seq_length=2000, threshold=1.0):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.threshold = threshold
        self.dna_seqs = []
        self.atac_signals = []

    def generate_atac_data(self):
        # generate pairs
        fa_name = 'resources/HC04.fasta'
        bw_name = 'resources/atac_bw/L21_1_R1.trim.srt.nodup.no_HC04_CM.tn5.pval0.01.500K.bfilt.bw'
        sp = SeqProcessor(fa_name)
        ep = EpigenProcessor(bw_name)
        pp = SeqEpigenPairPicker(sp, ep, seq_len=self.seq_length, threshold=self.threshold)
        pairs = pp.generate_whole_peak_pairs()
        for pair in pairs:
            self.dna_seqs.append(pair[0])
            self.atac_signals.append(pair[1])
        self.num_samples = len(pairs)

    def clear(self):
        self.dna_seqs = []
        self.atac_signals = []
        self.rna_signals = []

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        dna = self.dna_seqs[idx]
        atac = torch.from_numpy(self.atac_signals[idx])
        # Tokenize DNA
        encoding = self.tokenizer(dna, return_tensors='pt', truncation=True, max_length=self.seq_length)
        dna_batch = {'input_ids':encoding['input_ids'].squeeze(0),
                     'attention_mask':encoding['attention_mask'].squeeze(0)}
        return dna_batch, atac

if __name__ == '__main__':
    dataset = GenomicsDataset(seq_length=5000, threshold=0.1)
    dataset.all_expr()
    print(len(dataset.dna_seqs))
    print(len(dataset.atac_signals))
    print(len(dataset.rna_signals))
    print(dataset.dna_seqs[0])
    print(dataset.atac_signals[0])
    print(dataset.rna_signals[0])