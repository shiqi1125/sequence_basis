import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from peft import LoraConfig, get_peft_model

from GenomicDataset import *
from Module_Frozen import SequenceEncoder, ATACEncoder

from transformers import AutoTokenizer, AutoModel
from transformers import BatchEncoding

def info_nce_loss(dna_emb, atac_emb, temperature=0.1):
    B = dna_emb.size(0)
    dna_norm = F.normalize(dna_emb, p=2, dim=1)
    atac_norm = F.normalize(atac_emb, p=2, dim=1)
    sim_matrix = torch.matmul(dna_norm, atac_norm.T) / temperature
    labels = torch.arange(B, device=device)
    loss_d2a = F.cross_entropy(sim_matrix, labels)
    loss_a2d = F.cross_entropy(sim_matrix.T, labels)
    return (loss_d2a + loss_a2d) / 2

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    seq_encoder = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    def genomics_collate_fn(batch):
        # Unzip the batch
        input_ids_list, atac_list, label_list = zip(*batch)
        
        encodings = BatchEncoding({
            'input_ids': list(input_ids_list),
        })
        padded = tokenizer.pad(
            encodings,
            padding='longest',
            return_tensors='pt'
        )
        batched_input_ids = padded['input_ids']         # [B, L_max]
        batched_atac = torch.stack(list(atac_list), dim=0)
        batched_labels = torch.stack(list(label_list), dim=0)

        batched_input_ids = batched_input_ids.to(device, non_blocking=True)  
        batched_atac = batched_atac.to(device, non_blocking=True)  
        batched_labels = batched_labels.to(device, non_blocking=True)
        
        return batched_input_ids, batched_atac, batched_labels

    # init dataset and dataloader
    dataset = GenomicsDataset(tokenizer=tokenizer, seq_length=2000)
    dataset.all_expr()
    dataloader = DataLoader(dataset, 
                            batch_size=80, 
                            shuffle=True, 
                            collate_fn=genomics_collate_fn)

    # init DNABERT2 sequence encoder and apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["Wqkv", "dense", "gated_layers", "wo"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLASSIFICATION"
    )
    seq_model = SequenceEncoder(tokenizer=tokenizer, seq_encoder=seq_encoder, output_dim=128)
    #seq_encoder = get_peft_model(seq_model.seq_encoder, lora_config)
    seq_model.to(device)

    # init ATAC encoder
    atac_encoder = ATACEncoder(input_length=2000, emb_dim=128).to(device)

    if torch.cuda.device_count() > 1:
        seq_model = nn.DataParallel(seq_model)
        atac_encoder = nn.DataParallel(atac_encoder)
        # use Adam optimizer
        optimizer = torch.optim.Adam(
            list(seq_model.module.projection.parameters()) +
            list(atac_encoder.parameters()),
            lr=1e-4
        )
        # disable dropout
        seq_model.module.seq_encoder.eval()
    else:
        optimizer = torch.optim.Adam(
            list(seq_model.projection.parameters()) +
            list(atac_encoder.parameters()),
            lr=1e-4
        )
        # disable dropout
        seq_model.seq_encoder.eval()


    # # decoder head to reconstruct ATAC signal
    # atac_decoder = ATACDecoder(emb_dim=768, atac_dim=2000).to(device)

    # # use Adam optimizer
    # optimizer = torch.optim.Adam(
    #     list(seq_encoder.parameters()) +
    #     list(atac_encoder.parameters()) +
    #     list(atac_decoder.parameters()),
    #     lr=1e-4
    # )

    def save_model_state(model, save_path, epoch):
            torch.save({
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch
                        }, save_path)
            print("Model Saved.")

    log_name = "train_log"
    with open(log_name, 'w') as f:
        f.write("Log: \n")
    f.close()  

    # stage 1: contrastive learning and signal reconstruction
    num_epoch = 100
    for epoch in range(num_epoch):
        total_loss = 0.0
        for idx, (dna_ids, atac_signal, rng_cls) in enumerate(dataloader):
            dna_ids.to(device, non_blocking=True)
            atac_signal = atac_signal.to(device, non_blocking=True)
        
            # ffd
            dna_emb = seq_model(dna_ids)                    # [B, 128]
            #seq_emb = dna_outputs[0].mean(dim=1)                # [B,128]
            atac_emb = atac_encoder(atac_signal.unsqueeze(1))   # [B,128]

            # contrastive InfoNCE loss
            contras_loss = info_nce_loss(dna_emb, atac_emb, temperature=0.07)

            # # reconstruction loss
            # recons_signal = atac_decoder(seq_emb, atac_emb)        # [B,2000]
            # recons_loss = F.mse_loss(recons_signal, atac_signal)

            #final_loss = contras_loss + recons_loss
            optimizer.zero_grad()
            contras_loss.backward()
            optimizer.step()

            total_loss += contras_loss.item()
            if idx % 10 == 0:
                 print(f"Epoch {epoch} Iter {idx}: Train Loss={contras_loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        if epoch % 5 == 0:
            save_model_state(seq_model, 'seq_model.pth', epoch)
            save_model_state(atac_encoder, 'atac_encoder.pth', epoch)
            #save_model_state(atac_decoder, 'atac_decoder.pth', epoch)
        
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
        # print(f"contrastive loss {contras_loss.item():.4f}, reconstruction loss: {recons_loss.item():.4f}")
        with open(log_name, 'a') as f:
            f.write(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
            






