import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import pickle

from GenomicDataset import GenomicsDataset
from Module_Frozen import *

from transformers import AutoTokenizer, AutoModel
from transformers import BatchEncoding

from peft import PeftModel
from collections import OrderedDict

def pearson_corr(x, y):
    x = x.view(-1)
    y = y.view(-1)
    return torch.corrcoef(torch.stack((x, y)))[0, 1]

def genomics_collate_fn(batch):
    # Unzip the batch
    dna_list, atac_list, rna_list = zip(*batch)
    input_ids_list = [item['input_ids'] for item in dna_list]
    attention_mask_list = [item['attention_mask'] for item in dna_list]

    encodings = BatchEncoding({
        'input_ids': list(input_ids_list),
        'attention_mask':list(attention_mask_list)
    })
    padded = tokenizer.pad(
        encodings,
        padding='longest',
        return_tensors='pt'
    )
    batched_dna = {
        'input_ids': padded['input_ids'],    # [B, L_max]
        'attention_mask': padded['attention_mask']
    }
    batched_atac = torch.stack(list(atac_list), dim=0)
    batched_rna = torch.stack(list(rna_list), dim=0)
    
    return batched_dna, batched_atac, batched_rna

def clean_state(filename):
    ckpt = torch.load(filename)
    raw_state = ckpt["model_state_dict"]

    clean_state = OrderedDict()
    for k, v in raw_state.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        clean_state[new_key] = v
    
    return clean_state

if __name__ == '__main__':
    if not os.path.isfile('pred.cache'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoints_dir = 'checkpoints_loading'
        decoder_dir = "Model"
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        base_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

        tokenizer = AutoTokenizer.from_pretrained(checkpoints_dir)

        dataset = GenomicsDataset(tokenizer=tokenizer, seq_length=5000)
        dataset.chrom_expr('L021', 'HC04_A01')

        dna_encoder = PeftModel.from_pretrained(base_model, checkpoints_dir)
        atac_encoder = ATACEncoder2(768)
        decoder = CrossAttentionDecoder2()

        atac_encoder_path = os.path.join(checkpoints_dir, "atac_encoder.pth")
        decoder_path = os.path.join(decoder_dir, "decoder.pth")

        atac_encoder_state = clean_state(atac_encoder_path)
        decoder_state = clean_state(decoder_path)

        atac_encoder.load_state_dict(atac_encoder_state)
        decoder.load_state_dict(decoder_state)

        dna_encoder.to(device)
        atac_encoder.to(device)
        decoder.to(device)
        
        dna_encoder.eval()
        atac_encoder.eval()
        decoder.eval()

        # # 划分训练和验证集
        # train_size = int(0.9999 * len(dataset))
        # validation_size = len(dataset) - train_size
        # _, validation_dataset = torch.utils.data.random_split(
        #     dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(3)
        # )

        valid_loader = DataLoader(
            dataset, batch_size=80, shuffle=False, collate_fn=genomics_collate_fn
        )
        
        print("Starting prediction on validation set...")
        
        all_predictions = []
        all_labels = []
        all_dna_ids = []

        with torch.no_grad():
            for idx, (dna_batch, atac_batch, rna_batch) in enumerate(valid_loader):
                input_ids = dna_batch['input_ids']
                attention_mask = dna_batch['attention_mask']

                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                atac_batch = atac_batch.to(device, non_blocking=True)
                rna_batch = rna_batch.to(device, non_blocking=True)
                
                dna_emb = dna_encoder(input_ids=input_ids, attention_mask=attention_mask)
                dna_emb = dna_emb[0].mean(dim=1)
                atac_emb = atac_encoder(atac_batch)
                preds = decoder(atac_emb, dna_emb)

                all_predictions.append(preds.cpu())
                all_labels.append(rna_batch.cpu())
                #all_dna_ids.append(dna_ids.cpu())
            
        # 合并所有批次的结果
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        #all_dna_ids = torch.cat(all_dna_ids, dim=0)

        preds = []
        tgts = []
        for pred in all_predictions:
            preds.extend(pred.numpy())
        for tgt in all_labels:
            tgts.extend(tgt.numpy())

        del all_predictions
        del all_labels
        del all_dna_ids

        with open('pred.cache', 'wb') as f:
            pickle.dump(preds, f)
        f.close()

        with open('tgt.cache', 'wb') as f:
            pickle.dump(tgts, f)
        f.close()
    else:
        with open('pred.cache', 'rb') as f:
            preds = pickle.load(f)
        f.close()

        with open('tgt.cache', 'rb') as f:
            tgts = pickle.load(f)
        f.close()

    # ----------------------- 绘图代码 -----------------------
    # first_pred = all_predictions.numpy()  # 转为 numpy 数组
    # first_label = all_labels.numpy()      # 转为 numpy 数组
    x = np.arange(len(preds))           # x 轴为序列位置
    
    # 绘制预测值和实际值对比图
    # plt.figure(figsize=(14, 7))
    # plt.plot(x, preds, label='Prediction', color='blue', linewidth=2)
    # plt.plot(x, tgts, label='Actual Value', color='red', linewidth=2)
    # plt.xlabel('Position', fontsize=12)
    # plt.ylabel('Value', fontsize=12)
    # plt.title('Comparison of Predicted and Actual Values', fontsize=14)
    # plt.legend(fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.tight_layout()  # 调整布局
    # plt.savefig('prediction_actual_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()

    fig,axs = plt.subplots(2)
    axs[0].plot(tgts)
    axs[0].set_title('target')
    axs[1].plot(preds)
    axs[1].set_title('predict')
    plt.savefig('prediction.png', dpi=300, bbox_inches='tight')
