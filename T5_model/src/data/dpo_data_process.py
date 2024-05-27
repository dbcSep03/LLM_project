import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast
from fastparquet import write
import pandas as pd
import ujson
import re
import json
from tqdm.auto import tqdm
def whether_deleta_file(file: str) -> bool:
    if os.path.exists(file):
        print(f"File {file} already exists. Do you want to delete it? (y/n)")
        answer = input()
        if answer == "y":
            os.remove(file)
            return True
        else:
            return False


def write_single_parquet_file(df: pd.DataFrame, save_file_name: str) -> None:
    append = False
    if os.path.exists(save_file_name):
        append = True
    write(save_file_name, df, append=append, compression='GZIP')

def process_alphaca_gpt4(data_dir:str, output_dir:str, max_length: int=256, batch_size: int=256)->None:
    """
    将sft阶段训练2epoch的模型对alphaca_gpt4的数据进行处理, 生成的结果作为rejected
    """
    model = T5ForConditionalGeneration.from_pretrained('T5_model/sft_model/checkpoint-73294').cuda()
    model.eval()
    cur_rows = []
    tokenizer = PreTrainedTokenizerFast.from_pretrained('T5_model/pre_tokenizer/fast_tokenizer')
    batched_prompt = []
    batched_chosen = []
    data = json.load(open(data_dir, 'r', encoding='utf-8'))
    for line in tqdm(data,total=len(data)):
        if len(line['input'].strip()) > 0:
            prompt = line['instruction'] + "," + line['input']
        else:
            prompt = line['instruction']
        
        if len(prompt) > max_length or len(line['input']) > max_length:
            continue
        if len(prompt) == 0 or len(line['output']) == 0:
            continue
        batched_prompt.append(prompt)
        batched_chosen.append(line['output'])
        if len(batched_prompt)%batch_size == 0:
            input = [ f'{prompt}[EOS]'for prompt in batched_prompt]
            input_ids = tokenizer(input, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            outputs = model.generate(input_ids.input_ids.cuda(), max_length=max_length, num_beams=1, do_sample=False)
            batched_rejected = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            cur_rows.extend([{
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            } for prompt, chosen, rejected in zip(batched_prompt, batched_chosen, batched_rejected)])
            batched_prompt = []
            batched_chosen = []
        if len(cur_rows) > 1000:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(df, output_dir)
            cur_rows = []

    if len(batched_prompt) > 0:
        input = [ f'{prompt}[EOS]'for prompt in batched_prompt]
        input_ids = tokenizer(input, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        outputs = model.generate(input_ids.input_ids.cuda(), max_length=max_length, num_beams=1, do_sample=False)
        batched_rejected = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        cur_rows.extend([{
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        } for prompt, chosen, rejected in zip(batched_prompt, batched_chosen, batched_rejected)])
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(df, output_dir)




def replace_line(s: str) -> str:
    return re.sub('\\\\n', '\n', s)

def process_data_rlhf(data_dir:str, output_dir:str, max_length: int=256)->None:
    names = os.listdir(data_dir)
    names.remove('alpaca_gpt4_data_zh.json')
    cur_rows = []
    for name in names:
        if name.endswith('.json'):
            data = json.load(open(data_dir + name, 'r', encoding='utf-8'))
            for line in tqdm(data, total=len(data)):
                prompt, chosen, rejected = line['prompt'], line['chosen'], line['reject']
                if len(prompt) > max_length or len(line['chosen']) > max_length or len(line['reject']) > max_length:
                    continue
                if chosen.strip() == rejected.strip() or len(prompt) == 0 or len(chosen) == 0 or len(rejected) == 0:
                    continue
                cur_rows.append({
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected,
                })
                if len(cur_rows) >=10000:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(df, output_dir)
                    cur_rows = []
        if name.endswith('.parquet'):
            data = pd.read_parquet(data_dir + name)
            for index, row in tqdm(data.iterrows(), total = len(data)):
                prompt, chosen, rejected = row['prompt'], row['chosen'], row['rejected']
                if len(prompt) > max_length or len(chosen) > max_length or len(rejected) > max_length:
                    continue
                if chosen.strip() == rejected.strip() or len(prompt) == 0 or len(chosen) == 0 or len(rejected) == 0:
                    continue
                cur_rows.append({
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected,
                })
                if len(cur_rows) >= 10000:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(df, output_dir)
                    cur_rows = []
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(df, output_dir)
def test_parquet(dir):
    data = pd.read_parquet(dir)
    print(data.head())
    print(data.shape)
if __name__ == "__main__":
    output_dir = 'T5_model/data/processed/DPO/data.parquet'
    if os.path.exists(output_dir):
        assert whether_deleta_file(output_dir)
    process_alphaca_gpt4('T5_model/data/raw/RLHF/alpaca_gpt4_data_zh.json', output_dir)
    process_data_rlhf('T5_model/data/raw/RLHF/', output_dir)
    test_parquet(output_dir)