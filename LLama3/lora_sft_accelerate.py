# CUDA_VISIBLE_DEVICES="2,3" accelerate launch --multi_gpu --main_process_port 3288 LLama3/lora_sft_accelerate.py
from transformers import PreTrainedTokenizerFast
from src.models.model import LLamamodel
from src.models.config import modleConfig, SFTConfig, loraConfig
from src.models.lora import LoRA_model
from accelerate import load_checkpoint_and_dispatch, Accelerator
import torch
from tqdm.auto import tqdm
import pandas as pd
from src.models.LLamadataset import sftDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import wandb 
def pretrain_by_pytorch():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(SFTConfig.tokenizer_path)
    config = modleConfig(vocab_size = len(tokenizer), padding_idx=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    model = LLamamodel(config)
    model = load_checkpoint_and_dispatch(model, SFTConfig.model_path)
    print(model)
    lora_config = loraConfig()
    model = LoRA_model(model, lora_config.r, lora_config.alpha, lora_config.lora_name)
    print(model)
    total_nums = sum(param.numel() for param in model.model.parameters())
    trained_nums = sum(param.numel() for param in model.model.parameters() if param.requires_grad) 
    print(f"total parameters: {total_nums} trained parameters: {trained_nums} trained parameters: {trained_nums/total_nums} model size: {total_nums/1e9:.2f}B")
    data = pd.read_parquet(SFTConfig.dataset_path)

    train_dataset = sftDataset(data, config=config, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=SFTConfig.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader) * SFTConfig.epochs//SFTConfig.gradient_accumulation_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    accelerator = Accelerator(gradient_accumulation_steps=SFTConfig.gradient_accumulation_steps)
    model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader)

    for epoch in range(SFTConfig.epochs):
        loss = torch.Tensor([0])
        best_loss = 100000
        for input_id in tqdm(train_dataloader, total=len(train_dataloader),desc=f'epoch {epoch+1}/{SFTConfig.epochs}',disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                logits, _ = model(input_id)
                logits = logits[...,:-1,:].contiguous().view(-1, config.vocab_size)
                labels = input_id[...,1:].contiguous().view(-1)
                loss = criterion(logits, labels)/SFTConfig.gradient_accumulation_steps
                wandb.log({'loss': loss.item()})
                # print((loss*trainConfig.gradient_accumulation_steps).item())
                # print(accelerator.device)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        if loss.item() < best_loss:
            best_loss = loss.item()
            accelerator.wait_for_everyone()  # 必须加 让模型同步 
            accelerator.save_model(model, f'LLama3/checkpoints/LLama_sft_lora')
if __name__ == "__main__":
    wandb.init(project='LLama3',
            config={
                    'model': 'LLama',
                    'epochs': 8,
                    'gradient_accumulation_steps': 8,
                    'whether_multi_gpus': False,
            })

    pretrain_by_pytorch()