import pyBigWig
import numpy
from pyfaidx import Fasta

class SignalLoader():
    def __init__(self, fa_path, bw_path):
        self.seq_data = Fasta(fa_path)
        self.bw_file_data = pyBigWig.open(bw_path)
        self.chrom_data = self.bw_file_data.values

    def get_seq(self, chrom, start, end):
        seq = list(str(self.seq_data[chrom][start:end]))
        return seq
    
    def get_chrom_ids(self):
        return list(self.seq_data.keys())

    def get_chrom_sequence(self, chrom):
        seq_len = len(self.seq_data[chrom])
        seq = list(str(self.seq_data[chrom][:seq_len]))
        return seq
    
    def get_chrom_signal(self, chrom):
        seq_len = len(self.seq_data[chrom])
        return self.chrom_data(chrom, 0, seq_len)

    def get_signal(self, chrom, start, end):
        return self.chrom_data(chrom, start, end)
    
    def get_log_1_p_signal(self, chrom, start, end):
        return self.chrom_data(chrom, start, end), numpy.log1p(self.chrom_data(chrom, start, end))

    def close(self):
        self.seq_data.close()
        self.bw_file_data.close()

    def visualize(self, chrom, start, end):
        import matplotlib.pyplot as plt
        plt.plot(self.chrom_data(chrom, start, end), label='target')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Figure')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    data = SignalLoader('resources/tair10-3.fa', 'resources/peat.sorted.bw')
    #data.visualize('Chr1', 10000000, 20000000)
    # print(len(data.seq_data['Chr1']))
    # data.close()
    # print(data.get_log_1_p_signal('Chr1', 2632000, 2648000))
    bw = data.bw_file_data
    print(bw.chroms())
    print(len(data.seq_data['Chr1']))
    print(data.get_chrom_id()[0])
