from src.models.model import LLamamodel
from src.models.config import modleConfig, RlhfConfig
from transformers import PreTrainedTokenizerFast
from accelerate import load_checkpoint_and_dispatch
from src.models.LLamadataset import dpoDataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
import wandb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_logits(model, input_ids, label):
    logits,_ = model(input_ids)
    assert logits.shape[:-1] == label.shape
    label = label[:,1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (label != 0)
    per_toekn_logps = torch.gather(logits.log_softmax(-1), dim=2, index = label.unsqueeze(2)).squeeze(2)
    return (per_toekn_logps * loss_mask).sum(-1)/loss_mask.sum(-1)

def preference_loss(policy_chosen_logits, policy_rejected_logits, reference_chosen_logits, reference_rejected_logits, 
                    beta=0.2, label_smoothing=0.0):
    # beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
    # label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)

    chosen_loss = policy_chosen_logits - policy_rejected_logits
    rejected_loss = reference_chosen_logits - reference_rejected_logits
    
    logits = chosen_loss - rejected_loss
    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    chosen_rewards = (beta * (policy_chosen_logits - reference_chosen_logits)).detach().cpu().numpy().tolist()
    rejected_rewards = (beta * (policy_rejected_logits - reference_rejected_logits)).detach().cpu().numpy().tolist()

    return losses.mean(), chosen_rewards, rejected_rewards

def log_wandb(chosen_rewards, rejected_rewards):
    reward_accuracies = sum(1 for chosen, rejected in zip(chosen_rewards, rejected_rewards) if chosen > rejected)
    margins = sum(chosen - rejected for chosen, rejected in zip(chosen_rewards, rejected_rewards))
    chosen_rewards = np.mean(chosen_rewards)
    rejected_rewards = np.mean(rejected_rewards)
    # print(chosen_rewards, rejected_rewards, reward_accuracies, margins)
    wandb.log({'chosen_rewards': chosen_rewards, 'rejected_rewards': rejected_rewards, 'reward_accuracies': reward_accuracies, 'margins': margins})
def DPO():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(RlhfConfig.tokenizer_path)
    config = modleConfig(vocab_size = len(tokenizer), padding_idx=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    policy_model = LLamamodel(config).to(device)
    policy_model = load_checkpoint_and_dispatch(policy_model, RlhfConfig.model_path)

    reference_model = LLamamodel(config).to(device)
    reference_model = load_checkpoint_and_dispatch(reference_model, RlhfConfig.model_path)

    rlhf_data = pd.read_parquet(RlhfConfig.dataset_path)
    dataset = dpoDataset(rlhf_data, config, tokenizer)
    dataloader = DataLoader(dataset, batch_size=RlhfConfig.batch_size, shuffle=True,collate_fn=dataset.collate_fn)
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader) * RlhfConfig.epochs//RlhfConfig.gradient_accumulation_steps)
    for epoch in range(RlhfConfig.epochs):
        loss_all = 0
        chosen_rewards_all = []
        rejected_rewards_all = []
        for index, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'epoch {epoch+1}/{RlhfConfig.epochs}'):
            chsoen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            policy_chosen_logits = get_logits(policy_model, chsoen_input_ids, chosen_attention_mask)
            policy_rejected_logits = get_logits(policy_model, rejected_input_ids, rejected_attention_mask)
            with torch.no_grad():
                reference_chosen_logits = get_logits(reference_model, chsoen_input_ids, chosen_attention_mask)
                reference_rejected_logits = get_logits(reference_model,  rejected_input_ids, rejected_attention_mask)
            loss, chosen_rewards, rejected_rewards = preference_loss(policy_chosen_logits, policy_rejected_logits, reference_chosen_logits, reference_rejected_logits)
            loss_all += loss
            chosen_rewards_all += chosen_rewards
            rejected_rewards_all += rejected_rewards
            if(index + 1) % RlhfConfig.gradient_accumulation_steps == 0:
                loss_all.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                log_wandb(chosen_rewards_all, rejected_rewards_all)
                loss_all = 0
                chosen_rewards_all = []
                rejected_rewards_all = []
        torch.save(policy_model.state_dict(), 'LLama3/checkpoints/LLama_rlhf/model.pth')

        
        


if __name__ == "__main__":
    wandb.init(project='LLama3',
            config={
                    'model': 'LLama',
                    'epochs': 2,
                    'gradient_accumulation_steps': 8,
                    'whether_multi_gpus': False,
            })
    DPO()