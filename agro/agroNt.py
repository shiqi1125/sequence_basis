import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 加载进度条

# ========== 配置 ==========
MODEL_NAME = 'InstaDeepAI/agro-nucleotide-transformer-1b'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128  
NUM_EPOCHS = 100
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
            pooled = hidden_state.mean(dim=1)  
    return torch.stack(embeddings).mean(dim=0)

# ========== 加载数据 ==========
def load_data(csv_path, test_size=0.2):
    df = pd.read_csv(csv_path)
    sequences = df['sequence'].tolist()
    labels = df['label'].tolist()
    return train_test_split(sequences, labels, test_size=test_size, random_state=42, stratify=labels)

# ========== 训练函数 ==========
def train_model(model, classifier, train_loader, optimizer, criterion, tokenizer):
    model.eval()  # AgroNT 冻结
    classifier.train()
    total_loss = 0

    # 外层进度条：整个训练集
    for sequences, labels in tqdm(train_loader, desc="Training"):
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        embeddings_batch = []
        # 内层进度条：当前 batch 内每条序列
        for seq in tqdm(sequences, desc="Embedding Seq", leave=False):
            emb = get_sequence_embedding(seq, tokenizer, model)
            embeddings_batch.append(emb)
        embeddings_batch = torch.cat(embeddings_batch, dim=0).to(DEVICE)

        # 分类预测
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
    with torch.no_grad():
        for sequences, labels in tqdm(val_loader, desc="Evaluating"):
            labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(DEVICE)

            embeddings_batch = []
            for seq in tqdm(sequences, desc="Embedding Seq (Val)", leave=False):
                emb = get_sequence_embedding(seq, tokenizer, model)
                embeddings_batch.append(emb)
            embeddings_batch = torch.cat(embeddings_batch, dim=0).to(DEVICE)

            preds = classifier(embeddings_batch)
            predicted = (preds > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100
    return acc

# ========== 主函数 ==========
def main(csv_path):
    print("🚀 加载 tokenizer 和 AgroNT 模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    agro_nt = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    for param in agro_nt.parameters():
        param.requires_grad = False  # 冻结 AgroNT 权重

    classifier = MLPClassifier(embedding_dim=agro_nt.config.hidden_size).to(DEVICE)
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    print("📄 加载数据集...")
    train_seqs, val_seqs, train_labels, val_labels = load_data(csv_path)
    train_dataset = ExpressionDataset(train_seqs, train_labels)
    val_dataset = ExpressionDataset(val_seqs, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print("🏋️‍♂️ 开始训练...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(agro_nt, classifier, train_loader, optimizer, criterion, tokenizer)
        val_acc = evaluate_model(agro_nt, classifier, val_loader, tokenizer)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {train_loss:.4f} - Val Accuracy: {val_acc:.2f}%")

    torch.save(classifier.state_dict(), "mlp_classifier.pth")
    print("✅ 模型训练完成，分类器已保存为 mlp_classifier.pth")

# ========== 运行 ==========
if __name__ == "__main__":
    csv_file = "a/output_sequences_target.csv"  # CSV文件路径
    main(csv_file)

