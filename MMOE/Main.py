import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from scipy.stats import pearsonr

import numpy as np
import random

from Model import *
from Dataset import *

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    for dna, atac, rna, token_idx in tqdm(loader, desc="Train"):
        dna, atac, rna = dna.to(device), atac.to(device), rna.to(device)
        optimizer.zero_grad()
        pred = model(dna, atac, token_idx)
        loss = criterion(pred, rna)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        total_loss += loss.item() * dna.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, total_tokens, criterion, device):
    model.eval()
    sum_pred = np.zeros(total_tokens, dtype=np.float32)
    count_pred = np.zeros(total_tokens, dtype=np.int32)
    sum_true = np.zeros(total_tokens, dtype=np.float32)
    count_true = np.zeros(total_tokens, dtype=np.int32)
    total_loss = 0
    with torch.no_grad():
        for dna, atac, rna, token_idx in tqdm(loader, desc="Valid"):
            dna, atac, rna, token_idx = dna.to(device), atac.to(device), rna.to(device), token_idx.to(device)
            pred = model(dna, atac, token_idx)        # [B, token_cnt]
            loss = criterion(pred, rna)
            total_loss += loss.item() * dna.size(0)
            pred = pred.cpu().numpy()
            rna = rna.cpu().numpy()
            token_idx = token_idx.cpu().numpy()
            for i in range(pred.shape[0]):
                for j in range(pred.shape[1]):
                    gidx = token_idx[i, j]
                    sum_pred[gidx] += pred[i, j]
                    count_pred[gidx] += 1
                    sum_true[gidx] += rna[i, j]
                    count_true[gidx] += 1
    # Aggregate
    mean_pred = np.zeros_like(sum_pred)
    mean_true = np.zeros_like(sum_true)
    mask = count_pred > 0
    mean_pred[mask] = sum_pred[mask] / count_pred[mask]
    mean_true[mask] = sum_true[mask] / count_true[mask]
    # Genome-wide Pearson R
    corr = pearsonr(mean_pred[mask], mean_true[mask])[0]
    return corr, total_loss / len(loader.dataset)

def collate_fn(batch):
    dna, atac, rna, token_idx = zip(*batch)
    return (torch.stack(dna), torch.stack(atac), torch.stack(rna), torch.stack(token_idx))

def save_model_state(model, optimizer, epoch):
    os.makedirs('model_save', exist_ok=True)
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({
        "model_state": state_dict,
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch
    }, "model_save/model.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # manual seed
    torch.manual_seed(3)
    np.random.seed(3)
    random.seed(3)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3)

    batch_size = 32
    epochs = 300
    lr = 2e-4

    file_config, train_config = {}, {}
    file_config['dna'] = '../Data/ZM/B73-5.fa'
    file_config['rna'] = '../Data/ZM/RNA-Seq_map.pkl'
    file_config['atac'] = '../Data/ZM/atac_map.pkl'
    train_config['window_size'] = 128000
    train_config['token_len'] = 128
    train_config['stride'] = 16000
    train_config['holdout_chr'] = '10'

    logger, log_time = setup_logger()

    dataset = GenomicDataset(file_config=file_config, train_config=train_config)
    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset,
                                                    [train_size, valid_size],
                                                    generator=torch.Generator().manual_seed(3))
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              collate_fn=collate_fn,
                              pin_memory=True,
                              num_workers=8)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)

    model = GenomicsModel(d_model=128, nhead=8, nlayers=2, max_tokens=dataset.total_tokens).to(device)
    load_model = False
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Using {device_count} device with DataParallel")
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if load_model:
        checkpoint = torch.load("model_save/model.pth")
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=5)
    criterion = nn.MSELoss()

    best_corr = 0.1
    
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        corr, mse = eval_epoch(model, val_loader, dataset.total_tokens, criterion, device)
        
        logger.info(f"Epoch {epoch+1}: Train loss={loss:.4f}, Pearson R={corr:.4f}, Valid MSE={mse:.4f}")
        if corr > best_corr:
            best_corr = corr
            save_model_state(model, optimizer, epoch+1)
            logger.info(f"Model check point saved for epoch {epoch+1}")
        
