import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from GenomicDataset import GenomicsDataset
from Module_Frozen import SequenceEncoder, ATACEncoder, CrossAttentionDecoder

from transformers import AutoTokenizer, AutoModel
from transformers import BatchEncoding

def pearson_corr(x, y):
    # flatten to 1D
    x = x.view(-1)
    y = y.view(-1)

    return torch.corrcoef(torch.stack((x, y)))[0, 1]

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    seq_encoder = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    def genomics_collate_fn(batch):
        # Unzip the batch
        input_ids_list, atac_list, rna_list = zip(*batch)
        
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
        batched_rna = torch.stack(list(rna_list), dim=0)

        batched_input_ids = batched_input_ids.to(device, non_blocking=True)  
        batched_atac = batched_atac.to(device, non_blocking=True)  
        batched_rna = batched_rna.to(device, non_blocking=True)
        
        return batched_input_ids, batched_atac, batched_rna

    # init dataset and dataloader
    dataset = GenomicsDataset(tokenizer=tokenizer, seq_length=2000)

    # divide training and validation
    train_size = int(0.9 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset,
                                                                      [train_size, validation_size],
                                                                      generator=torch.Generator().manual_seed(3))

    train_loader = DataLoader(train_dataset, 
                            batch_size=80, 
                            shuffle=True, 
                            collate_fn=genomics_collate_fn)
    
    valid_loader = DataLoader(validation_dataset, 
                            batch_size=80, 
                            shuffle=True, 
                            collate_fn=genomics_collate_fn)

    seq_model = SequenceEncoder(tokenizer=tokenizer, seq_encoder=seq_encoder, output_dim=128)
    #seq_encoder = get_peft_model(seq_model.seq_encoder, lora_config)
    seq_model.to(device)

    # init ATAC encoder
    atac_encoder = ATACEncoder(input_length=2000, emb_dim=128).to(device)

    decoder = CrossAttentionDecoder()
    decoder.to(device)

    seq_model = nn.DataParallel(seq_model)
    atac_encoder = nn.DataParallel(atac_encoder)
    decoder = nn.DataParallel(decoder)

    if os.path.exists('Model'):
        # dna encoder contrastive learning result
        dna_encoder_checkpoint = torch.load('Model/seq_model.pth', map_location=device)
        seq_model.load_state_dict(dna_encoder_checkpoint['model_state_dict'])

        # atac encoder contrastive learning result
        atac_encoder_checkpoint = torch.load('Model/atac_encoder.pth', map_location=device)
        atac_encoder.load_state_dict(atac_encoder_checkpoint['model_state_dict'])

    # use Adam optimizer
    optimizer = torch.optim.Adam(
        list(decoder.parameters()),
        lr=1e-4
    )

    seq_model.eval()
    atac_encoder.eval()

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
        decoder.train()
        total_loss = 0.0
        for idx, (dna_ids, atac_signal, rna_signal) in enumerate(train_loader):
            dna_ids = dna_ids.to(device, non_blocking=True)
            atac_signal = atac_signal.to(device, non_blocking=True)
            rna_signal = rna_signal.to(device, non_blocking=True)
            optimizer.zero_grad()

            dna_emb = seq_model(dna_ids)                         # [B, 128]
            atac_emb = atac_encoder(atac_signal.unsqueeze(1))    # [B, 128]

            predicted_rna = decoder(atac_emb, dna_emb)

            # reconstruction loss
            loss = F.mse_loss(predicted_rna, rna_signal)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if idx % 10 == 0:
                 print(f"Epoch {epoch} Iter {idx}: Train Loss={loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)

        # validation
        decoder.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for dna_ids, atac_signal, rna_signal in valid_loader:
                dna_ids = dna_ids.to(device, non_blocking=True)
                atac_signal = atac_signal.to(device, non_blocking=True)
                rna_signal = rna_signal.to(device, non_blocking=True)
                dna_emb = seq_model(dna_ids)
                atac_emb = atac_encoder(atac_signal.unsqueeze(1))
                preds = decoder(atac_emb, dna_emb)
                valid_loss += F.mse_loss(preds, rna_signal)
        valid_loss /= len(valid_loader)

        if epoch % 5 == 0:
            save_model_state(decoder, 'decoder.pth', epoch)
            #save_model_state(atac_decoder, 'atac_decoder.pth', epoch)
        
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Valid loss: {valid_loss:.4f}")
        # print(f"contrastive loss {contras_loss.item():.4f}, reconstruction loss: {recons_loss.item():.4f}")
        with open(log_name, 'a') as f:
            f.writelines(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Valid loss: {valid_loss:.4f}")
            






