from torch.utils.data import Dataset
import datasets
import numpy as np
import torch
class T5Dataset(Dataset):
    def __init__(self,tokenizer, max_seq_length, train_data):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_length
        self.dataset = datasets.Dataset.from_parquet(train_data)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        """
        __getitem__ 出来的是文本 包括一些special token
        """
        prompt = self.dataset[item]["prompt"]
        response = self.dataset[item]["response"]
        max_length = self.max_seq_len - 5
        return {"prompt": prompt[:max_length] + "[EOS]", "response": response[:max_length] + "[EOS]"}

    def collate_fn(self, batch):
        """
        处理成 token的形式 先转换成int64 再转换成LongTensor
        """
        prompt = [info["prompt"] for info in batch]
        response = [info["response"] for info in batch]

        prompt = self.tokenizer(prompt, padding=True)
        response = self.tokenizer(response, padding=True)
        input_id = np.array(prompt["input_ids"], dtype=np.int64)
        response = np.array(response["input_ids"], dtype=np.int64)
        attention_mask = np.array(prompt["attention_mask"], dtype=np.int64)
        return {"input_ids": torch.LongTensor(input_id), "attention_mask": torch.LongTensor(attention_mask),
                "labels": torch.LongTensor(response)}