# 使用pytorch的DDP进行训练
import os 

from src.models.model import LLamamodel
from src.models.LLamadataset import LLamaDataset
from src.models.config import modleConfig, trainConfig

from transformers import PreTrainedTokenizerFast
from datasets import Dataset
from torch.utils.data import DataLoader

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from tqdm.auto import tqdm
import torch

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def pretrain_by_pytorch(rank, word_size):
    setup(rank, word_size)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(trainConfig.tokenizer_path)
    config = modleConfig(vocab_size = len(tokenizer), padding_idx=tokenizer.pad_token_id)
    model = LLamamodel(config).to(rank)
    ddp_model = DDP(model, device_ids=[rank])


    total_nums = sum(param.numel() for param in model.parameters())
    trained_nums = sum(param.numel() for param in model.parameters() if param.requires_grad)
    embedding_nums = sum(param.numel() for param in model.embedding.parameters())
    print(f"total parameters: {total_nums} trained parameters: {trained_nums} embedding parameters: {embedding_nums} model size: {total_nums/1e9:.2f}B")

    data = Dataset.from_parquet(trainConfig.dataset_path)
    dataset = LLamaDataset(data, trainConfig, tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=trainConfig.batch_size, shuffle=False, collate_fn=dataset.collate_fn, num_workers=4,sampler=DistributedSampler(dataset))   

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=5e-5)
    scheduler = CosineAnnealingLR(optimizer, len(train_dataloader) * trainConfig.epochs//trainConfig.gradient_accumulation_steps)


    for epoch in range(trainConfig.epochs):
        best_loss = 100000
        for idx, input_id in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'epoch {epoch+1}/{trainConfig.epochs}', disable=not rank==0):
            input_id = input_id['input_ids'].to(rank)
            logits, _ = ddp_model(input_id)
            logits = logits[...,:-1,:].contiguous().view(-1, config.vocab_size)
            labels = input_id[...,1:].contiguous().view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            if (idx + 1) % trainConfig.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        if loss.item() < best_loss and rank == 0:
            best_loss = loss.item()
            torch.save(model.state_dict(), f'LLama3/best_model_pytorch/model_best.checkpoint')
    cleanup()

def main(pretrain_by_pytorch, world_size):
    mp.spawn(pretrain_by_pytorch, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main(pretrain_by_pytorch, 2)