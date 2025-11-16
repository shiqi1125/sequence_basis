import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
#from torch.optim.lr_scheduler import ReduceLROnPlateau

from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from GenomicDataset import *
from Module_Frozen import *

from transformers import AutoTokenizer, AutoModel
from transformers import BatchEncoding
from accelerate import Accelerator, DistributedDataParallelKwargs

import os

def info_nce_loss(dna_emb, atac_emb, temperature=0.1):
    B = dna_emb.size(0)
    dna_norm = F.normalize(dna_emb, p=2, dim=1)
    atac_norm = F.normalize(atac_emb, p=2, dim=1)
    sim_matrix = torch.matmul(dna_norm, atac_norm.T) / temperature
    labels = torch.arange(B, device=sim_matrix.device)
    loss_d2a = F.cross_entropy(sim_matrix, labels)
    loss_a2d = F.cross_entropy(sim_matrix.T, labels)
    return (loss_d2a + loss_a2d) / 2

if __name__ == '__main__':
    # checkpoints dir for saving
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    use_checkpoints = False

    # checkpoints dir for loading
    checkpoints_dir = 'checkpoints_loading'
    if os.path.exists('checkpoints_loading'):
        use_checkpoints = True

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
            
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_ids = [0,1]
    primary_device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
    foundation_model_name = "zhihan1996/DNABERT-2-117M"
    if not use_checkpoints:
        tokenizer = AutoTokenizer.from_pretrained(foundation_model_name, trust_remote_code=True)
        # init DNABERT2 sequence encoder and apply LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["Wqkv", "dense", "gated_layers", "wo"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            inference_mode=False
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoints_dir)

    base_model = AutoModel.from_pretrained(foundation_model_name, trust_remote_code=True)

    def genomics_collate_fn(batch):
        # Unzip the batch
        dna_list, atac_list = zip(*batch)
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

        # batched_input_ids = batched_input_ids.to(device, non_blocking=True)  
        # batched_atac = batched_atac.to(device, non_blocking=True)  
        
        return batched_dna, batched_atac

    # init dataset and dataloader
    dataset = ATACDataset(tokenizer=tokenizer, seq_length=5000, threshold=0.1)
    dataset.generate_atac_data()
    dataloader = DataLoader(dataset, 
                            batch_size=32, 
                            shuffle=True,
                            collate_fn=genomics_collate_fn)

    # model init
    dna_encoder = PeftModel.from_pretrained(base_model, checkpoints_dir)
    atac_encoder = ATACEncoder2(768)
    if use_checkpoints:
        ckpt = torch.load(os.path.join(checkpoints_dir, "atac_encoder.pth"))
        raw_state = ckpt["model_state_dict"]

        clean_state = OrderedDict()
        for k, v in raw_state.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            clean_state[new_key] = v

        atac_encoder.load_state_dict(clean_state)

    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        list(dna_encoder.parameters()) +
        list(atac_encoder.parameters()),
        lr=1e-4
    )

    dna_encoder, atac_encoder, optimizer, dataloader = accelerator.prepare(dna_encoder, atac_encoder, optimizer, dataloader)

    #scheduler =  ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=5)

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

    model_checkpoint = "checkpoints"

    # stage 1: contrastive learning
    num_epoch = 100
    for epoch in range(num_epoch):
        dna_encoder.train()
        atac_encoder.train()
        total_loss = 0.0
        for idx, (dna_batch, atac_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            # embeddings
            input_ids = dna_batch['input_ids']
            attention_mask = dna_batch['attention_mask']

            dna_outputs = dna_encoder(input_ids=input_ids, attention_mask=attention_mask)
            dna_emb = dna_outputs[0].mean(dim=1)

            atac_emb = atac_encoder(atac_batch)
            # loss
            loss = info_nce_loss(dna_emb, atac_emb, temperature=0.07)
            # back propgation
            #loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            # logger
            total_loss += loss.item()
            if idx % 10 == 0 and accelerator.is_main_process:
                print(f"Epoch {epoch} Iter {idx}: Train Loss={loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        # checkpoint
        if epoch % 5 == 0 and accelerator.is_main_process:
            dna_encoder.module.save_pretrained(model_checkpoint)
            tokenizer.save_pretrained(model_checkpoint)
            save_model_state(atac_encoder, 'checkpoints/atac_encoder.pth', epoch)

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
            # print(f"contrastive loss {contras_loss.item():.4f}, reconstruction loss: {recons_loss.item():.4f}")
            with open(log_name, 'a') as f:
                f.writelines(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}.\n")
            f.close()
