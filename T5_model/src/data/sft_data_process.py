import os 
import re
import pandas as pd
import ujson
from fastparquet import write
from tqdm.auto import tqdm
punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：\n") 
def whether_deleta_file(file):
    if os.path.exists(file):
        print(f"File {file} already exists. Do you want to delete it? (y/n)")
        answer = input()
        if answer == "y":
            os.remove(file)
            return True
        else:
            return False

def remove_duplicate_punctuation(sentence:str)->str:
    """
    去除重复标点符号
    """
    sent = re.sub(r' |　', ',', sentence)
    new_sent, n = '', 0
    while(n<len(sent)):
        new_sent += sent[n]
        while (n < len(sent) -1) and (sent[n] in punctuation) and sent[n] in punctuation:
            n = n+1
        n = n + 1
    return new_sent 

def get_sentence_similarity(sent1:str, sent2:str)->float:
    """"
    计算句子相似度
    """    
    set_sent1 = set(sent1)
    set_sent2 = set(sent2)
    if (len(set_sent1) + len(set_sent2)) == 0:
        return 0
    return 2 * len(set_sent1 & set_sent2) / (len(set_sent1) + len(set_sent2))
def write_single_parquet_file(df: pd.DataFrame, save_file_name: str)->None:
    append = False
    if os.path.exists(save_file_name):
        append = True
    write(save_file_name, df, append=append, compression='GZIP')

def read_and_write_template(read_file: str, save_file_name: str, key_value: list[str], group_cnt: int=10000, special_f=None) ->None:
    with open(read_file, "r", encoding="utf-8") as f:
        cur_rows = []
        raw_line, keep_line = 0, 0
        total_len = len(f.readlines())
        f.seek(0)
        for line in tqdm(f,total=total_len):
            raw_line += 1
            line = ujson.loads(line)
            if special_f is not None and not special_f(line, key_value):
                continue
            if get_sentence_similarity(line["instruction"], line['input']) > 0.9:
                prompt = line["instruction"]
            else:
                prompt = line["instruction"] + " " + line['input']
                
            prompt = re.sub(r'\r', '', prompt)
            prompt = remove_duplicate_punctuation(prompt)
            output = re.sub(r'\r', '', line['output'])
            output = remove_duplicate_punctuation(output)

            if len(prompt) < 5 or len(line['output']) < 5:
                continue

            result = {"prompt": prompt, "response": output}
            cur_rows.append(result)
            keep_line+=1
            if len(cur_rows) > group_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(df, save_file_name)
                cur_rows = []
    if len(cur_rows)>0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(df, save_file_name)
            
        
def sft_data_process(input_dir, output_dir):
    output = os.path.join(output_dir, "sft_data.parquet")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output):
        assert whether_deleta_file(output)
    names = os.listdir(input_dir)

    def remove_data(line:dict, key_value: list[str]):
        instruction, input, output = line[key_value[0]], line[key_value[1]], line[key_value[2]]
        if '翻译' in instruction or 'translate' in output:
            return False
        if "表格" in instruction or '-----' in instruction or '-----' in output:
            return False
        return True
        
    for name in names:
        read_and_write_template(os.path.join(input_dir, name), output, key_value=["instruction", "input", 'output'], special_f=remove_data)
        
    
if __name__  == "__main__":
    sft_data_process(input_dir = "T5_model/data/raw/belle_sft", output_dir = "T5_model/data/processed/belle_sft")