import numpy as np
import pickle
import os
import logging
from datetime import datetime

BASE2IDX = {'A':0, 'C':1, 'G':2, 'T':3}

def deserialize_pickle(filename):
    data = None
    if os.path.isfile(filename):
        print(f"Deserializing {filename:s}")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        print(f"Deserialization {filename:s} done")
    f.close()
    return data

def dna_to_onehot(seq:str, window_size:int):
    seq = seq.upper()
    if len(seq) > window_size:
        seq = seq[:window_size]
    elif len(seq) < window_size:
        seq += "N" * (window_size - len(seq))  # pad with N

    assert len(seq) == window_size, f"Final sequence length is not 128000, got {len(seq)}"

    onehot = np.zeros((window_size, 4), dtype=np.int8)
    for i, base in enumerate(seq):
        if base in BASE2IDX:
            onehot[i, BASE2IDX[base]] = 1
    return onehot

def setup_logger(logdir="logs"):
    os.makedirs(logdir, exist_ok=True)
    log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(logdir, f"train_{log_time}.log")
    logger = logging.getLogger("GenomicsTrain")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_filename)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_time

