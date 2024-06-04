import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerFast
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
import json
import wandb
import pandas as pd

from src.models.model import LLamamodel
from src.models.LLamadataset import LLamaDataset
from src.models.config import modleConfig, trainConfig

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def pretrain_by_pytorch(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(trainConfig.tokenizer_path)
    config = modleConfig(vocab_size=len(tokenizer), padding_idx=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

    model = LLamamodel(config).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    total_nums = sum(param.numel() for param in model.parameters())
    trained_nums = sum(param.numel() for param in model.parameters() if param.requires_grad)
    embedding_nums = sum(param.numel() for param in model.embedding.parameters())
    if rank == 0:
        print(f"total parameters: {total_nums} trained parameters: {trained_nums} embedding parameters: {embedding_nums} model size: {total_nums/1e9:.2f}B")

    data = pd.read_parquet(trainConfig.dataset_path)
    print(len(data))
    dataset = LLamaDataset(data, trainConfig, tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(dataset, batch_size=trainConfig.batch_size // world_size, shuffle=False, collate_fn=dataset.collate_fn, sampler=sampler)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=5e-5)
    scheduler = CosineAnnealingLR(optimizer, len(train_dataloader) * trainConfig.epochs // trainConfig.gradient_accumulation_steps)

    scaler = torch.cuda.amp.GradScaler()

    loss_batch = 0
    for epoch in range(trainConfig.epochs):
        sampler.set_epoch(epoch)
        best_loss = float('inf')
        ddp_model.train()
        for idx, input_id in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'epoch {epoch+1}/{trainConfig.epochs}', disable=not rank == 0):
            input_id = input_id.to(rank)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, _ = ddp_model(input_id)
                logits = logits[...,:-1,:].contiguous().view(-1, config.vocab_size)
                labels = input_id[...,1:].contiguous().view(-1)
                loss = criterion(logits, labels)
            loss_batch += loss.item()
            scaler.scale(loss).backward()
            if (idx + 1) % trainConfig.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                # wandb.log({'loss': loss_batch / trainConfig.gradient_accumulation_steps})
                if rank == 0:
                    if (loss_batch / trainConfig.gradient_accumulation_steps) < best_loss:
                        best_loss = loss_batch / trainConfig.gradient_accumulation_steps
                        torch.save(model.state_dict(), f'LLama3/best_model_pytorch/model_best.checkpoint')
                        with open('LLama3/best_model_pytorch/loss.txt', 'w') as f:
                            json.dump({'loss': best_loss, 'epoch': epoch+1, 'idx': idx+1}, f)
                loss_batch = 0
    cleanup()

def main(world_size):
    mp.spawn(pretrain_by_pytorch, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    # wandb.init(project='LLama3',
    #     config={
    #             'model': 'LLama',
    #             'epochs': 8,
    #             'gradient_accumulation_steps': 8,
    #             'whether_multi_gpus': True,
    #     })
    main(2)
