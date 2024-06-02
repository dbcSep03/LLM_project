import os
import re
import time
import ujson
import pandas as pd
from fastparquet import write, ParquetFile
from tqdm.auto import tqdm
from opencc import OpenCC
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import json

punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：\n") # 标点符号
data_dir = 'LLama3/dataset/raw'
processed_file_dir = 'LLama3/dataset/sft'

def get_sentence_similarity(sentence1: str, sentence2: str) ->float:
    """
    计算两个句子的相似度
    相似度 等于2 * (len(set(A)&set(B)))/len(set(A)) + len(set(B))
    而且是中文 所以不用split(" ")到列表变成单词
    """
    set_a, set_b = set(sentence1) , set(sentence2)
    if len(set_a) + len(set_b) == 0:
        return 0
    return 2 * len(set_a&set_b)/(len(set_a) + len(set_b))


def whether_deleta_file(file_dir: str) ->bool:
    """
    判断是否删除已处理的文件
    """
    if os.path.exists(file_dir):
        choice = input(f"是否删除{file_dir}[y/n]")
        if choice == 'y' or choice == 'Y':
            os.remove(file_dir)
            print('文件已删除')
            return True
        return False

def remove_duplicate_punctuation(sentence: str) ->str:
    """
    删除重复的标点符号、重复的空格 将换行符变为特殊字符'\n'
    """
    sentence = re.sub(' |　', '，', sentence) 
    ans = ''
    n = len(sentence)
    p=0
    while p<n:
        ans += sentence[p]
        while (p+1<n) and (sentence[p+1 ] in punctuation) and (sentence[p] in punctuation): 
            p += 1
        p += 1
    return ans

def process_line_data(prompt, response, response_less_word: int=15) ->dict:
    """
    处理的流程
    先检测描述和标题是否相似
    如果相似则只用标题作为prompt 否则为标题+描述
    然后删除'\r'和重复的标点符号
    最后检测prompt至少长度为15 (考虑到一些任务 若自然语言推理任务 response可能很短 所以取消了response的长度限制)
    """
    prompt = prompt.replace('\r', '')
    remove_duplicate_punctuation(prompt)
    response = response.replace('\r', '')
    remove_duplicate_punctuation(response)
    if len(response) < response_less_word:
        return None
    else:
        return prompt + response + '<eos>'


def write_single_parquet_file(file_name: str, data: pd.DataFrame) ->None:
    """
    将数据写入parquet文件
    """
    append = False
    if os.path.exists(file_name):
        append = True
    write(file_name, data, append=append,compression='GZIP',)

def process_shareAI_data(data_dir: str='LLama3/dataset/raw/shareAI', save_file_name: str='LLama3/dataset/processed/shareAI.parquet', group_cnt: int=10000) ->None:
    """
    处理shareAI的数据
    """
    if os.path.exists(save_file_name):
        assert whether_deleta_file(save_file_name)
    names = os.listdir(data_dir)
    token_length_all = 0
    for name in names:
        token_length = 0
        read_file = os.path.join(data_dir, name)
        raw_line_cat = 0
        keep_line_cat = 0
        start_time = time.time()
        with open(read_file, 'r', encoding='utf-8') as f:
            cur_rows = []
            append = cur_rows.append
            total_len = len(f.readlines())
            f.seek(0)
            for line in tqdm(f, total=total_len):
                raw_line_cat += 1
                line = ujson.loads(line)['conversation'][0]
                input = line["human"]
                output = line['assistant']
                text = process_line_data(input,output)
                if text is None: continue
                
                append({'prompt':text})
                token_length += (len(text) - 5)
                keep_line_cat += 1

                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(save_file_name, df)
                    cur_rows = []
                    append = cur_rows.append
        if  len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(save_file_name, df)
            cur_rows = []
        end_time = time.time()
        token_length_all += token_length
        print(f"文件{read_file}处理完成, 用时{end_time - start_time}秒, 原始数据{raw_line_cat}条, 保留数据{keep_line_cat}条, token长度{token_length}")
    print(f"shareAI所有文件处理完成, token长度{token_length_all}")

def process_belle_data():
    """
    处理belle数据集
    """
    belle_dir = os.path.join(data_dir, 'belle_sft')
    file_names = os.listdir(belle_dir)
    save_file_name = os.path.join(processed_file_dir, 'belle.parquet')
    if os.path.exists(save_file_name):
        assert whether_deleta_file(save_file_name)

    
    group_cnt = 10000  
    token_length_all = 0
    
    for file in file_names:
        read_file = os.path.join(belle_dir, file)
        raw_line_cat = 0
        keep_line_cat = 0
        token_length = 0
        start_time = time.time()
        with open(read_file, 'r', encoding='utf-8') as f:
            cur_rows = []
            append = cur_rows.append

            total_len = len(f.readlines())
            f.seek(0)
            for line in tqdm(f, total=total_len):
                raw_line_cat += 1
                line = ujson.loads(line)
                title, desc, response = line["instruction"], line["input"], line["output"]
                # 删除翻译类任务
                if '翻译' in title or 'translate' in title.lower():
                    continue
                
                # 删除表格类任务
                if '表格' in title or '-----' in title or '-----' in response:
                    continue
                if get_sentence_similarity(title, desc) > 0.9:
                    prompt = title
                else:
                    prompt = title + desc
                
                text = process_line_data(prompt, response)
                if text is None: continue
                token_length += len(text) - 5
                append({'prompt':text})
                keep_line_cat += 1

                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(save_file_name, df)
                    cur_rows = []
                    append = cur_rows.append

        if  len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(save_file_name, df)
            cur_rows = []
        end_time = time.time()
        token_length_all += token_length
        print(f"文件{read_file}处理完成, 用时{end_time - start_time}秒, 原始数据{raw_line_cat}条, 保留数据{keep_line_cat}条, token长度{token_length}")
    print(f"belle所有文件处理完成, token长度{token_length_all}")

def process_CQIA_data():
    """
    处理COIG-CQIA数据集
    """
    data_dir = os.path.join(data_dir, 'COIG-CQIA','COIG-CQIA-full.jsonl')
    save_file_name = os.path.join(processed_file_dir, 'COIG-CQIA.parquet')
    if os.path.exists(save_file_name):
        assert whether_deleta_file(save_file_name)
    group_cnt = 10000
    token_length = 0
    raw_line_cat = 0
    keep_line_cat = 0
    with open(data_dir, 'r', encoding='utf-8') as f:
        cur_rows = []
        append = cur_rows.append
        for line in tqdm(f):
            raw_line_cat += 1
            line = ujson.loads(line)
            if len(line['input']) == 0:
                prompt = line['instruction']
            else:
                prompt = line['instruction'] + line['input']
            
            response = line['output']

            text = process_line_data(prompt, response)
            if text is None: continue
            token_length += len(text) - 5
            append({'prompt':text})
            keep_line_cat += 1

            if len(cur_rows) >= group_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(save_file_name, df)
                cur_rows = []
                append = cur_rows.append

        if  len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(save_file_name, df)
            cur_rows = []
    print(f"CQIA数据集处理完成, 原始数据{raw_line_cat}条, 保留数据{keep_line_cat}条, token长度{token_length}")

def process_alpaca_data():
    data_dir = os.path.join(data_dir, 'alpaca_data_zh_51k','alpaca_data_zh_51k.jsonl')
    save_file_name = os.path.join(processed_file_dir, 'alpaca.parquet')
    if os.path.exists(save_file_name):
        assert whether_deleta_file(save_file_name)
    group_cnt = 10000
    token_length = 0
    raw_line_cat = 0
    keep_line_cat = 0
    with open(data_dir, 'r', encoding='utf-8') as f:
        cur_rows = []
        append = cur_rows.append
        for line in tqdm(f):
            raw_line_cat += 1
            line = ujson.loads(line)
            if len(line['input']) == 0:
                prompt = line['instruction']
            else:
                prompt = line['instruction'] + line['input']
            
            response = line['output']

            text = process_line_data(prompt, response)
            if text is None: continue
            token_length += len(text) - 5
            append({'prompt':text})
            keep_line_cat += 1

            if len(cur_rows) >= group_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(save_file_name, df)
                cur_rows = []
                append = cur_rows.append

        if  len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(save_file_name, df)
            cur_rows = []
    print(f"alpaca数据集处理完成, 原始数据{raw_line_cat}条, 保留数据{keep_line_cat}条, token长度{token_length}")

def merge_dataset_as_single_file():
    """
    融合多个数据集为一个文件
    """
    save_file_name = os.path.join(processed_file_dir, 'sft_data.parquet')
    if os.path.exists(save_file_name):
        assert whether_deleta_file(save_file_name)
    
    parquet_files = [name  for name in os.listdir(processed_file_dir) if name.endswith('parquet') ]
    
    dataframes = [pd.read_parquet(os.path.join(processed_file_dir, file)) for file in parquet_files]
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    combined_df.to_parquet(save_file_name)

if __name__ == '__main__':
    process_shareAI_data()
    process_belle_data()
    process_CQIA_data()
    process_alpaca_data()
    merge_dataset_as_single_file()