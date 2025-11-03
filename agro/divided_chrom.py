import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score  # 🆕 计算 AUROC
import os
from datetime import datetime

# ========== 配置 ==========
MODEL_NAME = 'InstaDeepAI/agro-nucleotide-transformer-1b'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128  # 如果显存不足可调小
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
MAX_SEQ_LENGTH = 512  # AgroNT 最大支持长度
WINDOW_STRIDE = 256   # 滑动窗口步长

# ========== 数据集类 ==========
class ExpressionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        return seq, label

# ========== MLP 分类器 ==========
class MLPClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),  # 二分类
            nn.Sigmoid()
        )

    def forward(self, embeddings):
        return self.classifier(embeddings)

# ========== 提取序列embedding ==========
def get_sequence_embedding(seq, tokenizer, model):
    """
    使用 AgroNT 编码序列，支持滑动窗口
    """
    embeddings = []
    for i in range(0, len(seq), WINDOW_STRIDE):
        subseq = seq[i:i+MAX_SEQ_LENGTH]
        tokens = tokenizer(subseq,
                           return_tensors="pt",
                           padding="max_length",
                           truncation=True,
                           max_length=MAX_SEQ_LENGTH).to(DEVICE)
        with torch.no_grad():
            output = model(**tokens, output_hidden_states=True)
            hidden_state = output.hidden_states[-1]  # 最后一层 hidden state
            pooled = hidden_state.mean(dim=1)  # mean pooling
            embeddings.append(pooled)
    return torch.stack(embeddings).mean(dim=0)

# ========== 按染色体划分数据 ==========
def load_data_by_chrom(df, leave_out_chrom):
    """
    - 训练集：除leave_out_chrom外的所有数据
    - 验证集：仅leave_out_chrom的数据
    """
    train_df = df[df['chrom'] != leave_out_chrom]
    val_df = df[df['chrom'] == leave_out_chrom]

    train_sequences = train_df['full_sequence'].tolist()
    train_labels = train_df['target'].tolist()

    val_sequences = val_df['full_sequence'].tolist()
    val_labels = val_df['target'].tolist()

    return train_sequences, train_labels, val_sequences, val_labels

# ========== 训练函数 ==========
def train_model(model, classifier, train_loader, optimizer, criterion, tokenizer):
    model.eval()  # AgroNT 冻结
    classifier.train()
    total_loss = 0

    for sequences, labels in tqdm(train_loader, desc="Training"):
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        embeddings_batch = []
        for seq in tqdm(sequences, desc="Embedding Seq", leave=False):
            emb = get_sequence_embedding(seq, tokenizer, model)
            embeddings_batch.append(emb)
        embeddings_batch = torch.cat(embeddings_batch, dim=0).to(DEVICE)

        preds = classifier(embeddings_batch)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

# ========== 验证函数 ==========
def evaluate_model(model, classifier, val_loader, tokenizer):
    model.eval()
    classifier.eval()
    correct, total = 0, 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for sequences, labels in tqdm(val_loader, desc="Evaluating"):
            labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(DEVICE)

            embeddings_batch = []
            for seq in tqdm(sequences, desc="Embedding Seq (Val)", leave=False):
                emb = get_sequence_embedding(seq, tokenizer, model)
                embeddings_batch.append(emb)
            embeddings_batch = torch.cat(embeddings_batch, dim=0).to(DEVICE)

            probs = classifier(embeddings_batch)
            predicted = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100

    # 🆕 计算 AUROC
    try:
        auroc = roc_auc_score(all_labels, all_probs) * 100
    except ValueError:
        # 如果只有一个类别，会报错
        auroc = float('nan')

    return acc, auroc

# ========== 主函数 ==========
def main(csv_path, checkpoint_dir="./checkpoints"):
    print("🚀 加载 tokenizer 和 AgroNT 模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    agro_nt = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    for param in agro_nt.parameters():
        param.requires_grad = False  # 冻结 AgroNT 权重

    # 确保 checkpoint 目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 读取完整数据
    df = pd.read_csv(csv_path)

    # 循环每个染色体
    for leave_out_chrom in sorted(df['chrom'].unique()):
        print(f"\n🌱 当前轮次: 留出染色体 {leave_out_chrom} 作为验证集")
        train_seqs, train_labels, val_seqs, val_labels = load_data_by_chrom(df, leave_out_chrom)

        train_dataset = ExpressionDataset(train_seqs, train_labels)
        val_dataset = ExpressionDataset(val_seqs, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        classifier = MLPClassifier(embedding_dim=agro_nt.config.hidden_size).to(DEVICE)
        optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCELoss()

        # 日志文件 & 模型文件
        log_file = os.path.join(checkpoint_dir, f"log_chrom{leave_out_chrom}.txt")
        checkpoint = os.path.join(checkpoint_dir, f"mlp_classifier_chrom{leave_out_chrom}.pth")

        with open(log_file, "w") as f:
            for epoch in range(NUM_EPOCHS):
                print(f"🏋️‍♂️ Epoch [{epoch+1}/{NUM_EPOCHS}] for Chrom {leave_out_chrom}")
                train_loss = train_model(agro_nt, classifier, train_loader, optimizer, criterion, tokenizer)
                val_acc, val_auroc = evaluate_model(agro_nt, classifier, val_loader, tokenizer)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_str = (f"[{timestamp}] Chrom {leave_out_chrom} - Epoch [{epoch+1}/{NUM_EPOCHS}] "
                           f"- Train Loss: {train_loss:.4f} - Val Accuracy: {val_acc:.2f}% - Val AUROC: {val_auroc:.2f}%")
                print(log_str)
                f.write(log_str + "\n")

                # 保存断点
                torch.save(classifier.state_dict(), checkpoint)

        print(f"✅ 染色体 {leave_out_chrom} 模型训练完成，分类器已保存为 {checkpoint}")

    print("🎉 所有10轮训练完成")

# ========== 运行 ==========
if __name__ == "__main__":
    csv_file = "/home/miaoshiqi/a/zea_root_seq_target.csv"  # CSV路径
    main(csv_file, checkpoint_dir="/home/miaoshiqi/a/checkpoints")
