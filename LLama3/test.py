from datasets import Dataset
from torch.utils.data import DataLoader
from src.models.LLamadataset import LLamaDataset
from src.models.config import modleConfig, trainConfig
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
import pandas as pd
from datasets import Dataset
import os
import time
import dask.dataframe as dd
if __name__ == '__main__':
    start_time = time.time()
    # 通过指定engine 解决了第一个问题
    data = pd.read_parquet('LLama3/dataset/processed/pretrain_data.parquet')
    print(time.time() - start_time, len(data))

    # start_time = time.time()
    # data = dd.read_parquet('LLama3/dataset/processed/pretrain_data.parquet', columns=['input_ids'])
    # print(time.time() - start_time, len(data))
    



    # data.to_parquet('LLama3/dataset/processed/pretrain_data.parquet')
    # start_time = time.time()
    # data = pd.read_parquet('LLama3/dataset/processed/pretrain_data.parquet')
    # print(time.time() - start_time, len(data))
    # tokenizer = PreTrainedTokenizerFast.from_pretrained('LLama3/tokenizer/fast_tokenizer')
    # start_time = time.time()
    for idx in range(10):
        start_time = time.time()
        print(data['input_ids'][idx])
        print(time.time() - start_time)
        
    
   
        