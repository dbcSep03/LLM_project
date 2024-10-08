# 使用pytorch的DDP进行训练
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from src.models.model import LLamamodel
from src.models.LLamadataset import LLamaDataset
from src.models.config import modleConfig, trainConfig

from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR


from tqdm.auto import tqdm
import torch
import json
import wandb
import pandas as pd

def pretrain_by_pytorch():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(trainConfig.tokenizer_path)
    config = modleConfig(vocab_size = len(tokenizer), padding_idx=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

    model = LLamamodel(config).to('cuda')
    

    total_nums = sum(param.numel() for param in model.parameters())
    trained_nums = sum(param.numel() for param in model.parameters() if param.requires_grad)
    embedding_nums = sum(param.numel() for param in model.embedding.parameters())
    print(f"total parameters: {total_nums} trained parameters: {trained_nums} embedding parameters: {embedding_nums} model size: {total_nums/1e9:.2f}B")

    data = pd.read_parquet(trainConfig.dataset_path)
    print(len(data))
    dataset = LLamaDataset(data, trainConfig, tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=trainConfig.batch_size, shuffle=False, collate_fn=dataset.collate_fn, num_workers=4)   

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=5e-5)
    scheduler = CosineAnnealingLR(optimizer, len(train_dataloader) * trainConfig.epochs//trainConfig.gradient_accumulation_steps)

    loss_batch = 0
    for epoch in range(trainConfig.epochs):
        best_loss = 100000
        for idx, input_id in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'epoch {epoch+1}/{trainConfig.epochs}'):
            input_id = input_id.to('cuda')
            logits, _ = model(input_id)
            logits = logits[...,:-1,:].contiguous().view(-1, config.vocab_size)
            labels = input_id[...,1:].contiguous().view(-1)
            loss = criterion(logits, labels)
            loss_batch += loss.item()
            loss.backward()
            # wandb.log({'loss': loss.item()})
            if (idx + 1) % trainConfig.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                if (loss_batch/8) < best_loss:
                    best_loss = loss_batch/8
                    # torch.save(model.state_dict(), f'LLama3/best_model_pytorch/model_best.checkpoint')
                    with open('LLama3/best_model_pytorch/loss.txt', 'w') as f:
                        json.dump({'loss': best_loss, 'epoch': epoch+1, 'idx': idx+1}, f)
                loss_batch = 0



if __name__ == '__main__':
    # wandb.init(project='LLama3',
    #     config={
    #             'model': 'LLama',
    #             'epochs': 8,
    #             'gradient_accumulation_steps': 8,
    #             'whether_multi_gpus': True,
    #     })
    pretrain_by_pytorch()