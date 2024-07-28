import torch
from torch.utils.data import Dataset
import numpy as np
class LLamaDataset(Dataset):
    def __init__(self,data, config, tokenizer):
        super().__init__()
        self.data = data
        self.max_seq_length = config.seq_length
        self.tokenizer = tokenizer
        self.padding_id = tokenizer.pad_token_id
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        __getitem__ 选择对应的数据
        """
        input_id = self.data["input_ids"][index]
        return input_id

    def collate_fn(self, batch):
        """
        处理成批次 token的形式 先转换成int64 再转换成LongTensor
        """
        batch_np = np.array(batch, dtype=np.int64)
        input_ids = torch.LongTensor(batch_np)
        return input_ids
    
class sftDataset(Dataset):
    def __init__(self,data, config, tokenizer):
        super().__init__()
        self.data = data
        self.max_seq_length = config.seq_length
        self.tokenizer = tokenizer
        self.padding_id = tokenizer.pad_token_id
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        __getitem__ 选择对应的数据
        """
        input_id = self.data["input_ids"][index]
        return input_id
    
    def collate_fn(self, batch):
        """
        处理成批次 token的形式 先转换成int64 再转换成LongTensor
        """
        batch_np = np.array(batch, dtype=np.int64)
        input_ids = torch.LongTensor(batch_np)
        return input_ids

class dpoDataset(Dataset):
    def __init__(self,data, config, tokenizer):
        super().__init__()
        self.data = data
        self.max_seq_length = config.seq_length
        self.tokenizer = tokenizer
        self.padding_id = tokenizer.pad_token_id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        __getitem__ 选择对应的数据
        """
        prompt = self.data["prompt"][index]
        chosen = self.data["chosen"][index]
        rejected = self.data["reject"][index]
        return {
            "prompt": prompt,
            "chosen": chosen + '<EOS>',
            "rejected": rejected + '<EOS>'
        }
    def collate_fn(self, batch):
        prompts = [item["prompt"] for item in batch]
        chosens = [item["chosen"] for item in batch]
        rejecteds = [item["rejected"] for item in batch]

        tokenized_chosens = self.tokenizer(prompts, chosens, padding=True, truncation=True, max_length=self.max_seq_length, return_token_type_ids =True, return_tensors="pt")
        tokenized_rejecteds = self.tokenizer(prompts, rejecteds, padding=True, truncation=True, max_length=self.max_seq_length, return_token_type_ids =True, return_tensors="pt")
        chosen_input_ids = tokenized_chosens['input_ids']
        chosen_attention_mask = tokenized_chosens['token_type_ids']
        chosen_attention_mask = chosen_attention_mask * chosen_input_ids
        rejected_input_ids = tokenized_rejecteds['input_ids']
        rejected_attention_mask = tokenized_rejecteds['token_type_ids']
        rejected_attention_mask = rejected_attention_mask * rejected_input_ids

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask
        }
