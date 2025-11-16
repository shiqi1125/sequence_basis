import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist

from peft import PeftModel
from transformers import AutoTokenizer, AutoModel
from transformers import BatchEncoding
from accelerate import Accelerator, DistributedDataParallelKwargs

import os
from datetime import timedelta

from GenomicDataset import GenomicsDataset
from Module_Frozen import *
import Utils as utils

import gc

if __name__ == '__main__':
    dist.init_process_group(
    backend     = "nccl",
    init_method = "env://",
    timeout     = timedelta(hours=2),
    )
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision="fp16")
    device_ids = [0,1]
    primary_device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')

    checkpoints_dir = 'checkpoints_loading'
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    base_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    # load from first stage
    tokenizer = AutoTokenizer.from_pretrained(checkpoints_dir)
    #lora_config = LoraConfig.from_pretrained(checkpoints_dir)

    def genomics_collate_fn(batch):
        # Unzip the batch
        dna_list, atac_list, rna_list = zip(*batch)
        encodings = tokenizer(dna_list, truncation=True, max_length=5000, padding='longest', return_tensors='pt')
        batched_dna = {
            'input_ids': encodings['input_ids'].squeeze(0),    # [B, L_max]
            'attention_mask': encodings['attention_mask'].squeeze(0)
        }
        batched_atac = torch.stack(list(atac_list), dim=0)
        batched_rna = torch.stack(list(rna_list), dim=0)
        
        return batched_dna, batched_atac, batched_rna

    # init dataset and dataloader
    dataset = GenomicsDataset(tokenizer=tokenizer, seq_length=5000, threshold=0.1)
    dataset.all_expr()

    # divide training and validation
    train_size = int(0.9 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset,
                                                    [train_size, validation_size],
                                                    generator=torch.Generator().manual_seed(3))

    # keep a small validation dataset
    valid_dataset = Subset(valid_dataset, list(range(1000)))
    train_loader = DataLoader(train_dataset, 
                            batch_size=32, 
                            shuffle=True, 
                            num_workers=8,
                            pin_memory=True,
                            collate_fn=genomics_collate_fn)
    
    valid_loader = DataLoader(valid_dataset, 
                            batch_size=8, 
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            collate_fn=genomics_collate_fn)

    # module init
    dna_encoder = PeftModel.from_pretrained(base_model, checkpoints_dir)
    atac_encoder = ATACEncoder2(768)
    #dna_encoder.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'adapter_model.bin')))

    atac_state_file = os.path.join(checkpoints_dir, "atac_encoder.pth")
    atac_cs = utils.clean_state(atac_state_file)
    atac_encoder.load_state_dict(atac_cs)
    
    #dispersion = NBDispersion(init_r=0.1)

    decoder = CrossAttentionDecoder2()
    decoder_state_dir = "checkpoint_decoder"
    if os.path.exists(decoder_state_dir):
        decoder_state_file = os.path.join(decoder_state_dir, "decoder.pth")
        #dispersion_state_file = os.path.join(decoder_state_dir, "dispersion.pth")
        #dispersion_cs = utils.clean_state(dispersion_state_file)
        decoder_cs = utils.clean_state(decoder_state_file)
        decoder.load_state_dict(decoder_cs)
        #dispersion.load_state_dict(dispersion_cs)
        print("Load decoder state succeed.")

    # Freeze all parameters
    for param in dna_encoder.parameters():
        param.requires_grad = False

    for param in atac_encoder.parameters():
        param.requires_grad = False

    # use Adam optimizer
    optimizer = torch.optim.AdamW(
        list(decoder.parameters()),
        lr=1e-4
    )

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=5)

    dna_encoder, atac_encoder, decoder, \
        optimizer, scheduler, train_loader, valid_loader = accelerator.prepare(dna_encoder, 
                                                                                    atac_encoder, 
                                                                                    decoder,
                                                                                    optimizer, 
                                                                                    scheduler,
                                                                                    train_loader,
                                                                                    valid_loader)

    def save_model_state(model, optimizer, save_path, epoch):
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

    # freeze encoders
    dna_encoder.eval()
    atac_encoder.eval()

    # stage 1: contrastive learning and signal reconstruction
    num_epoch = 100
    best_corr = 0.0
    for epoch in range(num_epoch):
        # print("######### Train ############")
        decoder.train()
        total_loss = 0.0
        for idx, (dna_batch, atac_batch, rna_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = dna_batch['input_ids']
            attention_mask = dna_batch['attention_mask']

            dna_outputs = dna_encoder(input_ids=input_ids, attention_mask=attention_mask)

            dna_emb = dna_outputs[0].mean(dim=1)
            atac_emb = atac_encoder(atac_batch)
            predicted_rna = decoder(atac_emb, dna_emb)

            # reconstruction loss
            #base_disp = dispersion.module if hasattr(dispersion, "module") else dispersion
            #loss = utils.negative_binomial_loss(predicted_rna, rna_batch, base_disp.r)
            loss = F.mse_loss(predicted_rna, rna_batch)

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step(loss)

            del dna_batch, atac_batch, rna_batch, input_ids, attention_mask
            del dna_outputs, dna_emb, atac_emb, predicted_rna
            torch.cuda.empty_cache()
            gc.collect()

            total_loss += loss.item()
            if accelerator.is_main_process and idx % 10 == 0:
                 print(f"Epoch {epoch} Iter {idx}: Train Loss={loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader)

        print("######### Validation ############")
        corr = 0.0
        best_corr = 0.0
        valid_preds_corr, valid_tgts_corr = [], []
        decoder.eval()
        with torch.no_grad():
            n = 0
            for dna_batch, atac_batch, rna_batch in valid_loader:
                input_ids = dna_batch['input_ids']
                attention_mask = dna_batch['attention_mask']
                dna_outputs = dna_encoder(input_ids=input_ids, attention_mask=attention_mask)
                dna_emb = dna_outputs[0].mean(dim=1)
                atac_emb = atac_encoder(atac_batch)
                preds = decoder(atac_emb, dna_emb)

                valid_preds_corr.append(preds)
                valid_tgts_corr.append(rna_batch)

                del dna_batch, atac_batch, rna_batch, input_ids, attention_mask
                del dna_outputs, dna_emb, atac_emb, predicted_rna
                torch.cuda.empty_cache()
                gc.collect()

            corr = utils.pearson_corr(torch.cat(valid_preds_corr), torch.cat(valid_tgts_corr))
            del valid_tgts_corr
            del valid_preds_corr

            if accelerator.is_main_process:
                print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Valid corr: {corr.item():.4f}")
                with open(log_name, 'a') as f:
                    f.writelines(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Valid corr: {corr.item():.4f}\n")
                f.close()

                if corr > best_corr:
                    best_corr = corr
                    save_model_state(decoder, optimizer, "decoder.pth", epoch)
                    #save_model_state(dispersion, optimizer, "dispersion.pth", epoch)
                    with open(log_name, 'a') as f:
                        f.writelines(f"Model saved at epoch: {epoch+1}\n")
                    f.close()






