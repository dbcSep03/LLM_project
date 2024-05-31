import datasets
import os
from transformers import PreTrainedTokenizerFast
import pandas as pd
from tqdm.auto import tqdm
from fastparquet import write
def get_pretrain_data(data_all: datasets.Dataset, tokenizer: PreTrainedTokenizerFast, save_path: str = None): 

    def map_function(examples):
        input_id = tokenizer(examples['prompt'], padding=False, truncation=False)['input_ids']
        return {'input_ids': input_id}
    
    data_all = data_all.map(map_function, batched=True, remove_columns=['prompt'])

    token_ids_batch = []
    token_ids_all = []
    """
    形成batch
    """
    for input_id in tqdm(data_all['input_ids'], total=len(data_all)):
        token_ids_batch.extend(input_id)
        if len(token_ids_batch) > 512:
            token_ids_all.append({'input_ids': token_ids_batch[:512]})
            token_ids_batch = token_ids_batch[512:]
            if save_path is not None:
                token_ids_all_dataframe = pd.DataFrame(token_ids_all)
                write(save_path, token_ids_all_dataframe, append=os.path.exists(save_path))
    if len(token_ids_batch) > 0:
        token_ids_all.append({'input_ids': token_ids_batch})
    token_ids_all_dataframe = pd.DataFrame(token_ids_all)
    if save_path is not None:
        token_ids_all_dataframe.to_parquet(save_path)
        write(save_path, token_ids_all_dataframe, append=os.path.exists(save_path))
    data = datasets.Dataset.from_pandas(token_ids_all_dataframe)
    return data    
    

    
