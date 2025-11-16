import pyBigWig
import random
import numpy as np
import math
from pyfaidx import Fasta
import pickle
import os

class SeqProcessor():
    def __init__(self, file_name):
        self.seq_data = Fasta(file_name)

    def get_chrom_ids(self):
        return list(self.seq_data.keys())
    
    def get_chrom_sequence(self, chrom):
        seq_len = len(self.seq_data[chrom])
        seq = list(str(self.seq_data[chrom][:seq_len]))
        return seq
    
    def get_chrom_len(self, chrom):
        return len(self.seq_data[chrom])

    def get_chrom_subseq(self, chrom, start, end):
        return str(self.seq_data[chrom][start:end])

    def close(self):
        self.seq_data.close()

class EpigenProcessor():
    def __init__(self, file_name):
        try:
            self.epigen_file = pyBigWig.open(file_name)
            self.epigen_data = self.epigen_file.values
        except Exception:
            print("Cannot load bw file")

    def get_chroms(self):
        return self.epigen_file.chroms()
    
    def get_signal(self, chrom, start, end):
        return self.epigen_data(chrom, start, end)

    def close(self):
        self.epigen_file.close()

class CreateExpr():
    def __init__(self, sp, file_name, sample_name, use_cache=True):
        self.sp = sp
        self.sample_name = sample_name
        self.file_name = file_name
        self.cache_name = 'cache/' + '.'.join([sample_name, 'expr'])
        self.use_cache = use_cache
        if self.use_cache and not os.path.exists('cache'):
            os.mkdir('cache')

    def make_expr(self):
        if os.path.isfile(self.cache_name):
            return self.deserialize()
        from CSVProcessor import CSVProcessor
        chroms = self.sp.get_chrom_ids()
        cp = CSVProcessor(self.file_name)
        df = cp.read(self.sample_name).to_numpy()
        expr = {}
        for chrom in chroms:
            chrom_len = self.sp.get_chrom_len(chrom)
            expr_values = np.zeros(chrom_len, dtype=np.float32)
            for row in df:
                if row[0] == chrom:
                    expr_values[row[1]:row[2]] = np.float32(row[3])
            expr[chrom] = expr_values
        if self.use_cache:
            self.serialize(expr)
        return expr

    def serialize(self, data):
        if os.path.isfile(self.cache_name):
            print("Cache founded, serialization will not started until you delete the cache.")
        else:
            print("Serialization started..")
            with open(self.cache_name, 'wb') as f:
                pickle.dump(data, f)
            f.close()
            print("Serialization done.")

    def deserialize(self):
        if os.path.isfile(self.cache_name):
            print("Deserializing data..")
            with open(self.cache_name, 'rb') as f:
                data = pickle.load(f)
                print("Deserialization done.")
            f.close()
            return data

class SeqEpigenPairPicker():
    def __init__(self, 
                 sp, 
                 ep, 
                 seq_len=2000, 
                 threshold=1.0, 
                 use_cache=True):
        self.sp = sp
        self.ep = ep
        self.seq_len = seq_len
        self.threshold = threshold
        self.atac_cache_name = 'cache/' + '.'.join([str(self.seq_len), str(self.threshold), 'pkl'])
        self.use_cache = use_cache
        if self.use_cache and not os.path.exists('cache'):
            os.mkdir('cache')

    def validate_data(self):
        chroms = self.sp.get_chrom_ids()
        chroms_bw = self.ep.get_chroms()
        for chrom in chroms:
            cm_len_fa = len(self.sp.get_chrom_sequence(chrom))
            cm_len_bw = chroms_bw[chrom]
            print(f"Chrom id: {chrom} length in fa: {cm_len_fa} length in bw: {cm_len_bw}")

    def get_signals(self):
        chroms = self.sp.get_chrom_ids()
        signals = {}
        for chrom in chroms:
            chrom_len = len(self.sp.get_chrom_sequence(chrom))
            signal = self.ep.get_signal(chrom, 0, chrom_len)
            for i in range(len(signal)):
                if math.isnan(signal[i]):
                    signal[i] = 0.0
            signals[chrom] = signal
        return signals
    
    def remove_nan(self, signals):
        for i in range(len(signals)):
                if math.isnan(signals[i]):
                    signals[i] = 0.0

    # TODO: accelerate through multi-threading
    def generate_pairs_from_chrom(self, chrom_id):
        chrom_len = len(self.sp.get_chrom_sequence(chrom_id))
        end_idx = chrom_len - self.seq_len
        pairs = []
        for i in range(self.pick_num):
            seq_idx = random.randint(0, end_idx)  
            rand_seq = self.sp.get_chrom_subseq(chrom_id, seq_idx, seq_idx+self.seq_len)
            rand_signal = self.ep.get_signal(chrom_id, seq_idx, seq_idx+self.seq_len)
            for i in range(len(rand_signal)):
                if math.isnan(rand_signal[i]):
                    rand_signal[i] = 0.0
            rand_signal = np.array(rand_signal).astype(np.float32)
            rand_seq = rand_seq.replace('\x00', 'N')
            pair = (rand_seq, rand_signal)
            pairs.append(pair)
        return pairs

    def generate_peak_pair_by_chrom_id(self, chrom_id):
        chrom_len = len(self.sp.get_chrom_sequence(chrom_id))
        peak_range = self.seq_len
        signal = self.ep.get_signal(chrom_id, 0, chrom_len)
        self.remove_nan(signal)
        pairs = []
        for i in range(0, len(signal), peak_range):
            peaks = signal[i:i+peak_range]
            if sum(peaks) / len(peaks) > self.threshold and len(peaks) == peak_range:
                seq = self.sp.get_chrom_subseq(chrom_id, i, i+peak_range)
                seq = seq.replace('\x00', 'N')
                peaks = np.array(peaks).astype(np.float32)
                pairs.append((seq, peaks))
        return pairs

    def generate_whole_peak_pairs(self, use_cache=True):
        if use_cache and os.path.isfile(self.atac_cache_name):
            return self.deserialize(self.atac_cache_name)
        chroms = self.sp.get_chrom_ids()
        pairs = []
        for chrom in chroms:
            pair = self.generate_peak_pair_by_chrom_id(chrom)
            pairs.extend(pair)
        if use_cache:
            self.serialize(pairs, self.atac_cache_name)
        return pairs

    def generate_whole_expr_data(self, expr_data, npp=0.1, expr_threshold=1.0, use_cache=True, cache_name=""):
        if use_cache and cache_name != "" and os.path.isfile(cache_name):
            return self.deserialize(cache_name)
        chroms = self.sp.get_chrom_ids()
        pairs = []
        for chrom in chroms:
            pair = self.gen_expr_data_by_chrom(chrom, expr_data)
            pairs.extend(pair)
        if use_cache:
            self.serialize(pairs, cache_name)
        return pairs

    def gen_expr_data_by_chrom_id_cond(self, chrom_id, expr_data, npp, rna_threshold=0.1):
        chrom_len = len(self.sp.get_chrom_sequence(chrom_id))
        peak_range = self.seq_len
        signal = self.ep.get_signal(chrom_id, 0, chrom_len)
        self.remove_nan(signal)
        expr = expr_data[chrom_id]
        pairs = []
        for i in range(0, len(signal), peak_range):
            peaks = signal[i:i+peak_range]
            expr_value = expr[i:i+peak_range]
            if (len(peaks) == peak_range and len(expr_value) == peak_range):
                if (sum(expr_value) / len(expr_value) > rna_threshold or\
                sum(peaks) / len(peaks) > self.threshold):
                    seq = self.sp.get_chrom_subseq(chrom_id, i, i+peak_range)
                    seq = seq.replace('\x00', 'N')
                    peaks = np.array(peaks).astype(np.float32)
                    exprs = np.array(expr_value).astype(np.float32)
                    pairs.append([seq, peaks, exprs])
                # we want to use 10 percent of non peak data to catch to predict non peak data
                elif random.random() < npp:
                    seq = self.sp.get_chrom_subseq(chrom_id, i, i+peak_range)
                    seq = seq.replace('\x00', 'N')
                    peaks = np.array(peaks).astype(np.float32)
                    exprs = np.array(expr_value).astype(np.float32)
                    pairs.append([seq, peaks, exprs])
        return pairs

    def gen_expr_data_by_chrom(self, chrom_id, expr_data):
        chrom_len = len(self.sp.get_chrom_sequence(chrom_id))
        peak_range = self.seq_len
        signal = self.ep.get_signal(chrom_id, 0, chrom_len)
        self.remove_nan(signal)
        expr = expr_data[chrom_id]
        pairs = []
        for i in range(0, len(signal), peak_range):
            peaks = signal[i:i+peak_range]
            expr_value = expr[i:i+peak_range]
            if len(peaks) == peak_range and len(expr_value) == peak_range:
                seq = self.sp.get_chrom_subseq(chrom_id, i, i+peak_range)
                seq = seq.replace('\x00', 'N')
                peaks = np.array(peaks).astype(np.float32)
                exprs = np.array(expr_value).astype(np.float32)
                pairs.append([seq, peaks, exprs])
        return pairs

    def get_peaks_pairs(self):
        if os.path.isfile(self.atac_cache_name):
            print("Load data from cache.")
            pairs = self.deserialize()
        else:
            print("Cache not found, creating dataset.")
            pairs = self.generate_whole_peak_pairs()
            if self.use_cache:
                print("Serializing..")
                self.serialize(pairs, self.atac_cache_name)
                print("Serialization done.")
        return pairs

    def serialize(self, data, cache_name):
        if os.path.isfile(cache_name):
            print("Cache founded, serialization will not started until you delete the cache.")
        else:
            print("Serialization started..")
            with open(cache_name, 'wb') as f:
                pickle.dump(data, f)
            f.close()
            print("Serialization done.")

    def deserialize(self, cache_name):
        if os.path.isfile(cache_name):
            print("Deserializing data..")
            with open(cache_name, 'rb') as f:
                pairs = pickle.load(f)
                print("Deserialization done.")
            f.close()
            return pairs

    def visualize(self, data):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2)
        axs[0].plot(data[1])
        axs[0].set_title('atac')
        axs[1].plot(data[2])
        axs[1].set_title('rna')
        plt.show()

    def close(self):
        self.ep.close()
        self.sp.close()

if __name__ == '__main__':
    fa_name = 'resources/HC04.fasta'
    bw_name = 'resources/atac_bw/L21_1_R1.trim.srt.nodup.no_HC04_CM.tn5.pval0.01.500K.bfilt.bw'
    sp = SeqProcessor(fa_name)
    ep = EpigenProcessor(bw_name)
    pp = SeqEpigenPairPicker(sp, ep, seq_len=5000, threshold=1.0)
    #pp.validate_data()
    ce = CreateExpr(sp=sp, file_name='resources/cotton_rnaseq_true.csv',
                    sample_name='L021',
                    use_cache=True)
    expr_data = ce.make_expr()
    cache_name = "cache/L021.whole"
    data = pp.generate_whole_expr_data(expr_data, use_cache=True, cache_name=cache_name)
    pp.serialize(data, cache_name)
    test = pp.deserialize(cache_name)

    print(len(test))
    print(len(test[0]))
    pp.visualize(test[100])

    pp.close()