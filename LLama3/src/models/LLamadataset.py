import torch
from torch.utils.data import Dataset

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
        print(len(input_id))
        return input_id

    def collate_fn(self, batch):
        """
        处理成批次 token的形式 先转换成int64 再转换成LongTensor
        """
        input_ids = torch.LongTensor(batch)
        return input_ids
    
        