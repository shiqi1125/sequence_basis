import pickle
import os
import numpy
import torch
from GenomicDataset import GenomicDataset, Tokenize

class DataProcessor():
    def __init__(self):
        self.token_ids = []
        self.token_id_map = {}
        self.tgt_signal = []
        self.tgt_signal_map = {}
        if not os.path.exists('cache'):
            os.mkdir('cache')

    def prepare_data_from_random_sequence(self, seq_len, k=6):
        from RandSeqBuilder import RandSeqGenerator
        rsg = RandSeqGenerator(seq_len)
        sequence = rsg.genome_sequence
        self.tgt_signal = rsg.signal
        tk = Tokenize(k, sequence)
        self.token_ids = tk.tokens
        self.tgt_signal = self.tgt_signal[:len(self.token_ids)]

    def get_token_ids_by_chrom(self, chrom, start, end):
        return self.token_id_map[chrom][start:end]
    
    def get_tgt_signal_by_chrom(self, chrom, start, end):
        return self.tgt_signal_map[chrom][start:end]

    def add_data_from_bigwig(self, fa_file, bw_file, k=6):
        from SignalLoader import SignalLoader
        sl = SignalLoader(fa_file, bw_file)

        bw_file_name = bw_file.split('/')[-1]
        cache_name = [bw_file_name, str(k), 'token_ids','pkl']
        cache_name = 'cache/' + '.'.join(cache_name)
        
        if os.path.isfile(cache_name):
            print("Token cache founded, deserializing.")
            with open(cache_name, 'rb') as f:
                self.token_id_map = pickle.load(f)
                print("Tokenization complete.")
        else:
            print("Couldn't find token cache, serializing...")
            print("It could take some time...")
            chrom_ids = sl.get_chrom_ids()
            for chrom in chrom_ids:
                sequence = sl.get_chrom_sequence(chrom)
                tk = Tokenize(k, sequence)
                token_ids = tk.tokens
                self.token_id_map[chrom] = token_ids
            # serializing
            with open(cache_name, 'wb') as f:
                pickle.dump(self.token_id_map, f)
            print("Tokenization complete...")
                    
        for chrom_id, token_ids in self.token_id_map.items():
            chrom_signal = sl.get_chrom_signal(chrom_id)
            # drop the last k signals
            chrom_signal = chrom_signal[:len(token_ids)]
            self.tgt_signal_map[chrom_id] = chrom_signal

        sl.close()

    def generate_one_chrom_dataset(self, window_size, overlap, chrom, valid_ratio, log1p_scale=True, seed=42):
        self.token_ids = self.token_id_map[chrom]
        self.tgt_signal = self.tgt_signal_map[chrom]
        ts = self.tgt_signal
        if log1p_scale:
            ts = numpy.log1p(self.tgt_signal)
        # tensornization
        tgt_signal_tensors = torch.FloatTensor(ts)
        token_ids_tensors = torch.tensor(self.token_ids, dtype=torch.long)

        dataset = GenomicDataset(token_ids_tensors, tgt_signal_tensors, window_size, overlap)
        valid_size = int(valid_ratio * len(dataset))
        train_size = len(dataset) - valid_size
        train_dataset, validation_dataset = torch.utils.data.random_split(dataset,
                                                                 [train_size, valid_size], 
                                                                 generator=torch.Generator().manual_seed(seed))
        return train_dataset, validation_dataset

    def generate_dataset(self, window_size, overlap, training_chroms, valid_chroms, valid_ratio, log1p_scale=True, seed=42):
        for chrom, token_ids in self.token_id_map.items():
            if chrom not in valid_chroms and chrom in training_chroms:
                self.token_ids = self.token_ids + token_ids
                self.tgt_signal = self.tgt_signal + self.tgt_signal_map[chrom]
        
        ts = self.tgt_signal
        if log1p_scale:
            ts = numpy.log1p(self.tgt_signal)

        # tensornization
        tgt_signal_tensors = torch.FloatTensor(ts)
        token_ids_tensors = torch.tensor(self.token_ids, dtype=torch.long)
        # generate dataset
        train_dataset = GenomicDataset(token_ids_tensors, tgt_signal_tensors, window_size, overlap)
        valid_tokens, valid_signals = [], []
        for chrom in valid_chroms:
            valid_tokens = valid_tokens + self.token_id_map[chrom]
            valid_signals = valid_signals + self.tgt_signal_map[chrom]
        valid_signals = torch.FloatTensor(valid_signals)
        valid_tokens = torch.tensor(valid_tokens, dtype=torch.long)
        full_valid_dataset = GenomicDataset(valid_tokens, valid_signals, window_size, overlap)
        valid_size = int(valid_ratio * len(full_valid_dataset))
        drop_size = len(full_valid_dataset) - valid_size
        drop, validation_dataset = torch.utils.data.random_split(full_valid_dataset,
                                                                 [drop_size, valid_size], 
                                                                 generator=torch.Generator().manual_seed(seed))
        return train_dataset, validation_dataset
            

if __name__ == '__main__':
    dp = DataProcessor()
    dp.add_data_from_bigwig('resources/tair10-3.fa', 'resources/peat.sorted.bw')