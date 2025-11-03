import torch
import random
import numpy as np

class RandSeqGenerator():
    def __init__(self, seq_len):
        bases = ['A','C','G','T']
        # random sequence builder
        self.genome_sequence = ''.join(random.choice(bases) for _ in range(seq_len))
        self.signal = np.random.randn(seq_len)
        #self.signal = torch.randn(seq_len, dtype=torch.float)

    def print(self):
        print(f"Sequence: {self.genome_sequence}.\n")
        print(f"Signal: {self.signal}.\n")
    
    def visualize(self):
        import matplotlib.pyplot as plt
        plt.plot(self.signal, label='signal')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Figure')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    signal = RandSeqGenerator(10000)
    signal.visualize()

