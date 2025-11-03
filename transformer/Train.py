import torch
import os
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

class Trainer():
    def __init__(
            self,
            train_dataset=None,
            valid_dataset=None,
            train_from_scratch=False,
            window_size=1024,
            overlap=512,
            batch_size=4,
            lr=1e-4,
            num_workers=4,
            epochs=5,
            weight_decay=1e-2,
            d_model=128,
            n_head=8,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.1,
            k=6,
            local_window=128,
            save_path='Model.pth',
            device=torch.device('cuda'),
            logger='log.txt'
            ):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_from_scratch = train_from_scratch
        self.window_size = window_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.k = k
        self.local_window = local_window
        self.save_path = save_path
        self.device = device
        self.logger = logger

    def weighted_mse(self, pred, tgt, threshold=4.0, weight_factor=5.0):
        error = (pred - tgt) ** 2
        weights = torch.ones_like(tgt)
        weights[tgt > threshold] = weight_factor
        weighted_error = weights * error
        return torch.mean(weighted_error)

    def dynamic_weight_mse(self, pred, tgt, threshold=2.0):
        error = (pred - tgt) ** 2
        weights = 1 + torch.clamp(tgt - threshold, min=0)
        weighted_error = weights * error
        return torch.mean(weighted_error)

    def exponential_weight_mse(self, pred, tgt, threshold=2.0, gamma=0.5):
        error = (pred - tgt) ** 2
        weights = torch.ones_like(tgt)
        weight_factor = torch.exp(gamma * torch.clamp(tgt - threshold, min=0))
        weights = weights * weight_factor
        weighted_error = weights * error
        return torch.mean(weighted_error)

    def negative_binomial_loss(self, pred, tgt, r=1.0):
        tgt = torch.expm1(tgt)
        pred= torch.expm1(pred)
        r_tensor = torch.tensor(r, dtype=tgt.dtype, device=tgt.device)
        term1 = torch.lgamma(r_tensor + tgt) - torch.lgamma(tgt + 1) - torch.lgamma(r_tensor)
        term2 = r_tensor * torch.log(r_tensor / (r_tensor + pred))
        term3 = tgt * torch.log(pred / (r_tensor + pred) + 1e-8)
        nll = -(term1 + term2 + term3)
        return torch.mean(nll)

    def PseudoPoissonKL(self, pred, target):
        return torch.mean(target * torch.log((target + 1e-10)/(pred + 1e-10)) + pred - target)

    def criterion(self, pred, tgt):
        # return F.mse_loss(pred, tgt)
        # Huber loss
        # return F.smooth_l1_loss(input=pred,
        #                         target=tgt)
        # return self.PseudoPoissonKL(pred, tgt)
        #return F.poisson_nll_loss(pred, tgt, log_input=False, reduction='mean')
        return self.weighted_mse(pred, tgt)

    def pearson_corr(self, x, y):
        # flatten to 1D
        x = x.view(-1)
        y = y.view(-1)

        return torch.corrcoef(torch.stack((x, y)))[0, 1]

    def train(self):
        '''
        start training process
        '''
        # divide training and validation dataset
        # train_size = int(0.8 * len(self.dataset))
        # validation_size = len(self.dataset) - train_size
        # train_dataset, validation_dataset = torch.utils.data.random_split(self.dataset,
        #                                                                   [train_size, validation_size],
        #                                                                   generator=torch.Generator().manual_seed(42))

        # # instead of random split, use the first 20 percent as validation dataset
        # validation_size = int(0.2 * len(self.dataset))
        # validation_dataset = Subset(self.dataset, list(range(validation_size)))
        # train_dataset = Subset(self.dataset, list(range(validation_size, len(self.dataset))))

        # poisson_loss = nn.PoissonNLLLoss(log_input=False, reduction='mean')

        # initialize dataloader
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=True)
        validation_loader = DataLoader(self.valid_dataset,
                                       batch_size=self.batch_size,
                                       num_workers=self.num_workers,
                                       pin_memory=True)

        # initialize model
        from GenomicTransformModel import GenomicTransformModel
        model = GenomicTransformModel(vocab_size=4 ** self.k,
                                    d_model=self.d_model,
                                    nhead=self.n_head,
                                    num_layers=self.num_layers,
                                    dim_feedforward=self.dim_feedforward,
                                    dropout=self.dropout,
                                    window_size=self.window_size,
                                    local_window=self.local_window,
                                    gradient_checkpoint=True)

        model.to(self.device)

        if not self.train_from_scratch:
            if os.path.exists(self.save_path):
                checkpoint = torch.load(self.save_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print("######## train from saved #########")
            else:
                print("Cannot file Model.pth, train from scratch.")
        else:
            print("######## train from scratch #########")

        # initialize optimizer
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scaler = GradScaler(device='cuda')

        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=5)

        train_losses = []
        validation_losses = []
        validation_corrs = []
        eps = []

        def save_model_state():
            torch.save({
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch
                        }, self.save_path)
            print("Model Saved.")

        # best_loss = 1000.0
        best_corr = 0.0
        # training
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            corr = 0.0
            train_preds_corr, train_tgts_corr = [], []
            for idx, (seq_batch, tgt_batch) in enumerate(train_loader):
                seq_batch = seq_batch.to(self.device, non_blocking=True)
                tgt_batch = tgt_batch.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                with autocast('cuda'):
                    preds = model(seq_batch)
                    loss = self.criterion(preds, tgt_batch)
                # backpropgation
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_preds_corr.append(preds.detach())
                train_tgts_corr.append(tgt_batch.detach())
                total_loss += loss.item()
                if idx % 50 == 0:
                    print(f"Epoch {epoch} Iter {idx}: Train Loss={loss.item():.4f}")
            avg_loss = total_loss / len(train_loader)
            corr = self.pearson_corr(torch.cat(train_preds_corr), torch.cat(train_tgts_corr))
            del train_preds_corr
            del train_tgts_corr
            # validation
            model.eval()
            valid_loss = 0.0
            valid_preds_corr, valid_tgts_corr = [], []
            with torch.no_grad():
                for seq_batch, tgt_batch in validation_loader:
                    seq_batch = seq_batch.to(self.device, non_blocking=True)
                    tgt_batch = tgt_batch.to(self.device, non_blocking=True)
                    preds = model(seq_batch)
                    valid_loss += self.criterion(preds, tgt_batch).item()
                    valid_preds_corr.append(preds)
                    valid_tgts_corr.append(tgt_batch)
            valid_loss /= len(validation_loader)
            valid_corr = self.pearson_corr(torch.cat(valid_preds_corr), torch.cat(valid_tgts_corr))
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(valid_corr)
            print(f"Epoch {epoch}: average training loss = {avg_loss:.4f}, \
                  validation loss = {valid_loss:.4f}, current lr = {current_lr:.6f},\
                    train corr = {corr:.4f}, validation corr = {valid_corr:.4f}")
            # logger
            train_losses.append(avg_loss)
            validation_losses.append(valid_loss)
            validation_corrs.append(valid_corr)
            eps.append(epoch)
            if best_corr < valid_corr:
                best_corr = valid_corr
                with open(self.logger, 'w') as f:
                    for i in range(len(train_losses)):
                        f.write(f"Epoch: {eps[i]}, Train loss: {train_losses[i]:.4f}, Learning rate: {current_lr}\
                                Validation loss: {validation_losses[i]:.4f}, Validation corrs: {validation_corrs[i]:.4f}\n")
                    print("Log generated.")
                    f.close()
                save_model_state()
        save_model_state()

if __name__ == '__main__':
    tr = Trainer(random_seq_mode=False, epochs=5)
    tr.train()