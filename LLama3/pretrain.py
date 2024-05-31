import os
# 使用40系显卡
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from src.models.model import LLamamodel
from src.models.LLamadataset import LLamaDataset
from src.models.config import modleConfig, trainConfig

from transformers import PreTrainedTokenizerFast
from datasets import Dataset
from torch.utils.data import DataLoader

import torch
from torch import optim,nn
from accelerate import Accelerator

from tqdm.auto import tqdm
import wandb

def pretrain_by_pytorch():
    #加载tokenizer 和 model 并统计相关信息
    tokenizer = PreTrainedTokenizerFast.from_pretrained(trainConfig.tokenizer_path)
    config = modleConfig(vocab_size = tokenizer.vocab_size, padding_idx=tokenizer.pad_token_id)
    model = LLamamodel(config)
    total_nums = sum(param.numel() for param in model.parameters())
    trained_nums = sum(param.numel() for param in model.parameters() if param.requires_grad) 
    embedding_nums = sum(param.numel() for param in model.embedding.parameters())
    print(f"total parameters: {total_nums} trained parameters: {trained_nums} embedding parameters: {embedding_nums} model size: {total_nums/1e9:.2f}B")
    
    # 训练集 测试集
    data = Dataset.from_parquet(trainConfig.dataset_path)
    
    train_dataset = LLamaDataset(data, trainConfig, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader) * trainConfig.epochs//trainConfig.gradient_accumulation_steps)
    criterion = nn.CrossEntropyLoss()

    accelerator = Accelerator(gradient_accumulation_steps=trainConfig.gradient_accumulation_steps)
    model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader)

    for epoch in range(trainConfig.epochs):
        loss = torch.Tensor([0])
        best_loss = 100000
        for input_id in tqdm(train_dataloader, total=len(train_dataloader),desc=f'epoch {epoch+1}/{trainConfig.epochs}',disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                input_id = input_id['input_ids']
                logits, _ = model(input_id)
                logits = logits[...,:-1,:].contiguous().view(-1, config.vocab_size)
                labels = input_id[...,1:].contiguous().view(-1)
                loss = criterion(logits, labels)/trainConfig.gradient_accumulation_steps
                wandb.log({'loss': loss.item()})
                # print((loss*trainConfig.gradient_accumulation_steps).item())
                # print(accelerator.device)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        if loss.item() < best_loss:
            best_loss = loss.item()
            accelerator.save_checkpoint(f'LLama3/checkpoints/LLama_pretrain.pth')
        
        

if __name__ == '__main__':
    wandb.init(project='LLama3',
            config={
                    'model': 'LLama',
                    'epochs': 8,
                    'gradient_accumulation_steps': 8,
                    'whether_multi_gpus': False,
            })

    pretrain_by_pytorch()