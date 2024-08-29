import os 
import threading
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"] = "1"
import numpy as np
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from peft import TaskType
from torch.nn import CrossEntropyLoss
import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from torch.optim import AdamW
from accelerate import Accelerator
from itertools import chain

from tqdm.auto import tqdm

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba
from rouge_chinese import Rouge
import wandb 
def token_map(exapmle):
    input = exapmle['input']
    output = exapmle['target']
    input = "<|im_start|>问：\n{}\n答：\n".format(input.strip().replace('答：', '').strip())
    output = "{}\n<|im_end|>".format(output.strip())
    input_len = len(tokenizer(input, add_special_tokens=False)['input_ids'])

    results = tokenizer(text=input+output, add_special_tokens=False)
    results["labels"] = [-100]*input_len + results["input_ids"][input_len:]
    return {'input_ids': results['input_ids'], 'labels': results['labels']}

# def data_collator(examples):
#     input = [example['input_ids'] for example in examples]
#     labels = [example['labels'] for example in examples]
#     max_len = min(max(len(x) for x in input), tokenizer.model_max_length)
#     print(max_len)
#     input = [x + [tokenizer.im_end_id]*(max_len-len(x)) for x in input]
#     labels = [x + [-100]*(max_len-len(x)) for x in labels]
#     return {
#         "input_ids": torch.LongTensor(input),
#         "labels": torch.LongTensor(labels)
#     }
#     block_size = 128


def group_texts(examples):
    block_size = 128
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}

    total_length = len(concatenated_examples["input_ids"])

    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size + 1) * block_size
    # Split by chunks of max_len.
    result = {}
    for k, t in concatenated_examples.items():
        if total_length > len(t):
            if k == "input_ids":
                t = t + [tokenizer.im_end_id] * (total_length - len(t))
            else:
                t = t + [-100] * (total_length - len(t))

        truncs = [t[i : i + block_size] for i in range(0, total_length, block_size)]
        result[k] = truncs

    return result

def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            hypothesis = ' '.join(hypothesis)
            if not hypothesis:
                hypothesis = "-"
            scores = rouge.get_scores(hypothesis, ' '.join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

if __name__ == "__main__":
    
    run = wandb.init(project="NER_model")
    tokenizer =AutoTokenizer.from_pretrained("qwen-7b", trust_remote_code=True)
    train_data = datasets.Dataset.from_json('data/train.json')
    train_data = train_data.map(token_map,remove_columns=["input", "target","answer_choices","task_dataset",'sample_id','task_type'])
    train_data = train_data.map(group_texts, batched=True)
    
    test_data = datasets.Dataset.from_json('data/test.json')
    test_data = test_data.map(token_map,remove_columns=["input", "target","answer_choices","task_dataset",'sample_id','task_type'])
    test_data = test_data.map(group_texts, batched=True)

    tokenizer.pad_token = tokenizer.decode(tokenizer.im_end_id)
    
    config = LoraConfig(r=8,task_type=TaskType.CAUSAL_LM,target_modules=["c_attn"], lora_dropout=0.1, bias="none")
    model = AutoModelForCausalLM.from_pretrained("qwen-7b", trust_remote_code=True)
    # model.half()
    print(model)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_data = DataLoader(train_data, batch_size=1, collate_fn=data_collator)
    test_data = DataLoader(test_data, batch_size=1, collate_fn=data_collator)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = CrossEntropyLoss(ignore_index=-100)
    train_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {train_param}")
    accelerator = Accelerator(gradient_accumulation_steps=8)
    epochs = 6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,len(train_data)*epochs/8)
    model, optimizer, train_data,test_data, scheduler, = accelerator.prepare(model, optimizer, train_data, test_data, scheduler)
    for epoch in range(epochs):

        for batch in tqdm(train_data, desc=f"Epoch {epoch}",total=len(train_data)):
            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                labels = batch['labels']
                outputs = model(input_ids=input_ids, labels=labels)
                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                accelerator.backward(loss)
                if threading.current_thread().getName() == "MainThread":
                    print(f"train_loss: {loss.item()}")
                    run.log({"train_loss": loss.item()})
                loss_main = loss.item() 
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        model.eval()
        for batch in tqdm(test_data, desc=f"Epoch {epoch}",total=len(test_data)):
            with torch.no_grad():
                input_ids = batch['input_ids']
                labels = batch['labels']
                outputs = model(input_ids=input_ids, labels=labels)
                # loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                # if threading.current_thread().getName() == "MainThread":
                #     print(f"test_loss: {loss.item()}")
                preds = outputs.logits.argmax(-1)
                score_dict = compute_metrics((preds, labels))
                if threading.current_thread().getName() == "MainThread":
                    print(score_dict)
                    run.log(score_dict)
    output_dir = "pretrain_model"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")
    