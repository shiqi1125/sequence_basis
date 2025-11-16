import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict

from GenomicDataset import GenomicsDataset
from Module_Frozen import *
from peft import PeftModel

from transformers import AutoTokenizer, AutoModel
from transformers import BatchEncoding

from accelerate import Accelerator, DistributedDataParallelKwargs

def pearson_corr(x, y):
    # flatten to 1D
    x = x.view(-1)
    y = y.view(-1)

    return torch.corrcoef(torch.stack((x, y)))[0, 1]

if __name__ == '__main__':
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
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

    # init dataset and dataloader
    dataset = GenomicsDataset(tokenizer=tokenizer, seq_length=5000, threshold=0.1)
    dataset.all_expr()

    # divide training and validation
    train_size = int(0.9 * len(dataset))
    validation_si32ze = len(dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset,
                                                                      [train_size, validation_size],
                                                                      generator=torch.Generator().manual_seed(3))

    train_loader = DataLoader(train_dataset, 
                            batch_size=32, 
                            shuffle=True, 
                            collate_fn=genomics_collate_fn)
    
    valid_loader = DataLoader(validation_dataset, 
                            batch_size=32, 
                            shuffle=True, 
                            collate_fn=genomics_collate_fn)

    # module init
    dna_encoder = PeftModel.from_pretrained(base_model, checkpoints_dir)
    #dna_encoder.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'adapter_model.bin')))

    # some fix for saving error...
    ckpt = torch.load(os.path.join(checkpoints_dir, "atac_encoder.pth"))
    raw_state = ckpt["model_state_dict"]

    clean_state = OrderedDict()
    for k, v in raw_state.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        clean_state[new_key] = v
    
    baseline = DNABaseline()

    # Freeze all parameters
    for param in dna_encoder.parameters():
        param.requires_grad = False

    # use Adam optimizer
    optimizer = torch.optim.AdamW(
        list(baseline.parameters()),
        lr=1e-4
    )

    dna_encoder, baseline, optimizer, train_loader, valid_loader = accelerator.prepare(dna_encoder, 
                                                                                    baseline,
                                                                                    optimizer, 
                                                                                    train_loader,
                                                                                    valid_loader)

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

    # freeze encoders
    dna_encoder.eval()

    # stage 1: contrastive learning and signal reconstruction
    num_epoch = 100
    for epoch in range(num_epoch):
        baseline.train()
        total_loss = 0.0
        for idx, (dna_batch, atac_batch, rna_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = dna_batch['input_ids']
            attention_mask = dna_batch['attention_mask']

            # input_ids = input_ids.to(primary_device, non_blocking=True)
            # attention_mask = attention_mask.to(primary_device, non_blocking=True)
            # atac_batch = atac_batch.to(primary_device, non_blocking=True)
            # rna_batch = rna_batch.to(primary_device, non_blocking=True)

            dna_outputs = dna_encoder(input_ids=input_ids, attention_mask=attention_mask)

            dna_emb = dna_outputs[0].mean(dim=1)
            predicted_rna = baseline(dna_emb)

            # reconstruction loss
            loss = F.mse_loss(predicted_rna, rna_batch)

            accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item()
            if accelerator.is_main_process and idx % 10 == 0:
                 print(f"Epoch {epoch} Iter {idx}: Train Loss={loss.item():.4f}")

        if accelerator.is_main_process:
            # validation
            baseline.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for dna_batch, atac_batch, rna_batch in valid_loader:
                    input_ids = dna_batch['input_ids']
                    attention_mask = dna_batch['attention_mask']
                    dna_outputs = dna_encoder(input_ids=input_ids, attention_mask=attention_mask)
                    dna_emb = dna_outputs[0].mean(dim=1)
                    preds = baseline(dna_emb)

                    valid_loss += F.mse_loss(preds, rna_batch)

            valid_loss /= len(valid_loader)

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Valid loss: {valid_loss:.4f}")
            with open(log_name, 'a') as f:
                f.writelines(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Valid loss: {valid_loss:.4f}")

        if accelerator.is_main_process and epoch % 5 == 0:
            save_model_state(baseline, 'baseline.pth', epoch)
            






