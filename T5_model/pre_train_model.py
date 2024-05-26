import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from src.models.T5_model import T5model
from src.models.T5Dataset import T5Dataset
from transformers import PreTrainedTokenizerFast
from config import TokenizerConfig, ModelConfig
from torch.utils.data import DataLoader
from config import TrainConfig
import datasets
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
import wandb
import torch.nn as nn
import torch
def train():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TokenizerConfig.fast_tokenizer)
    model = T5model(len(tokenizer.get_vocab()),
                    d_model=ModelConfig.d_model, num_heads=ModelConfig.num_heads, d_kv=ModelConfig.d_kv,
                    d_ff=ModelConfig.d_ff,num_encoder_layers=ModelConfig.num_layers,
                    num_decoder_layers=ModelConfig.num_decoder_layers,dropout=0.1,
                    bos_id=tokenizer.bos_token_id, padding_id=tokenizer.pad_token_id)
    
    data = datasets.Dataset.from_parquet(TrainConfig.data)
    data = data.train_test_split(test_size=0.01)
    train_dataset = T5Dataset(tokenizer, TrainConfig.max_seq_length, data["train"])
    train_dataloader = DataLoader(train_dataset, batch_size=TrainConfig.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_dataset = T5Dataset(tokenizer, TrainConfig.max_seq_length, data["test"]) 
    val_dataloader = DataLoader(val_dataset, batch_size=TrainConfig.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

    optimizer = Adam(model.parameters(), lr=TrainConfig.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, TrainConfig.max_epochs * len(train_dataloader) // TrainConfig.gradient_accumulation_steps, eta_min=0)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    loss = 0
    for epoch in range(TrainConfig.max_epochs):
        model.train()
        for index, batch in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch+1}/{TrainConfig.max_epochs}",total=len(train_dataloader)):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()
            loss = model.train_net(input_ids, attention_mask, labels=labels, criterion=criterion)
            if (index % TrainConfig.gradient_accumulation_steps == 0):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            wandb.log({"train_loss": loss.item()})
        model.eval()
        with torch.no_grad():
            for index, batch in tqdm(enumerate(val_dataloader),desc=f"Epoch {epoch+1}/{TrainConfig.max_epochs}",total=len(val_dataloader)):
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                labels = batch["labels"].cuda()
                loss = model.train_net(input_ids, attention_mask, labels=labels, criterion=criterion)
                wandb.log({"eval_loss": loss.item()})
        torch.save(model.state_dict(), f"{TrainConfig.output_dir_pytorch}/model_{epoch}.pt")


if __name__ == "__main__":
    wandb.login()
    wandb.init(project="T5-training-pytorch",
               config= {
                   "learning_rate": TrainConfig.learning_rate,
               })
    train()