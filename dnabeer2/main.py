import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
import numpy as np

# -------------------- 参数设置 --------------------
TSV_FILE = "sequence_expression_pairs_full.tsv"
MODEL_NAME = "zhihan1996/DNABERT-2-117M"
BATCH_SIZE = 32
EPOCHS = 100
LR = 5e-5
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "dnabert2_regression_model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- 数据集定义 --------------------
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        enc = self.tokenizer(seq, truncation=True, padding="max_length",
                             max_length=self.max_length, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in enc.items()}
        item['labels'] = torch.tensor(label, dtype=torch.float)
        return item

# -------------------- 读取数据 --------------------
df = pd.read_csv(TSV_FILE, sep="\t")
train_df = df[df['split']=='train']
val_df   = df[df['split']=='val']
test_df  = df[df['split']=='test']

# -------------------- 加载模型和分词器 --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1, trust_remote_code=True)

# 多 GPU 支持
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model.to(DEVICE)

# -------------------- DataLoader --------------------
train_dataset = SequenceDataset(train_df['sequence'], train_df['tpm_log'], tokenizer, MAX_LEN)
val_dataset   = SequenceDataset(val_df['sequence'], val_df['tpm_log'], tokenizer, MAX_LEN)
test_dataset  = SequenceDataset(test_df['sequence'], test_df['tpm_log'], tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------- 优化器 --------------------
optimizer = AdamW(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()

# -------------------- 训练 --------------------
best_val_r = -1

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for batch in train_loader_tqdm:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.view(-1)  # 确保是 1 维
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(labels)
        train_loader_tqdm.set_postfix(loss=loss.item())
    train_loss /= len(train_dataset)

    # -------------------- 验证 --------------------
    model.eval()
    val_true, val_pred = [], []
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
    with torch.no_grad():
        for batch in val_loader_tqdm:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.view(-1)  # 确保是 1 维
            val_pred.extend(logits.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    val_r = pearsonr(val_true, val_pred)[0]
    val_mse = mean_squared_error(val_true, val_pred)
    val_r2  = r2_score(val_true, val_pred)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Pearson r: {val_r:.4f} - MSE: {val_mse:.4f} - R2: {val_r2:.4f}")

    # -------------------- 保存最优模型 --------------------
    if val_r > best_val_r:
        best_val_r = val_r
        if isinstance(model, torch.nn.DataParallel):
            model.module.save_pretrained(OUTPUT_DIR)
        else:
            model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Saved best model.")

# -------------------- 测试 --------------------
model.eval()
test_true, test_pred = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.view(-1)  
        test_pred.extend(logits.cpu().numpy())
        test_true.extend(labels.cpu().numpy())

r_test = pearsonr(test_true, test_pred)[0]
mse_test = mean_squared_error(test_true, test_pred)
r2_test  = r2_score(test_true, test_pred)
print(f"Test Pearson r: {r_test:.4f} - MSE: {mse_test:.4f} - R2: {r2_test:.4f}")
