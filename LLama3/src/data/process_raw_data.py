"""
处理后的文件都先保存为parquet的格式
"""
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
processed_file_dir = 'LLama3/dataset/processed'
en_punctuation = ",().!;:"
zh_punctuation = "，（）。！；："


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

def convert_en_punctuation_to_zh_punct(sentence: str) -> str:
    '''
    将句子中的英文标点替换文中文标点
    '''
    n = len(zh_punctuation)
    for i in range(n):
        sentence = sentence.replace(en_punctuation[i], zh_punctuation[i])
    return sentence

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


def write_single_parquet_file(file_name: str, data: pd.DataFrame) ->None:
    """
    将数据写入parquet文件
    """
    append = False
    if os.path.exists(file_name):
        append = True
    write(file_name, data, append=append,compression='GZIP',)


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

def process_ruozhiba_data(data_dir: str='LLama3/dataset/raw/ruozhiba', save_file_name: str='LLama3/dataset/processed/ruozhiba.parquet', group_cnt: int=10000) ->None:
    if os.path.exists(save_file_name):
        assert whether_deleta_file(save_file_name)
    read_file = os.path.join(data_dir, 'ruozhiba_qa.json')
    raw_line_cat = 0
    keep_line_cat = 0
    start_time = time.time()
    data = json.load(open(read_file, 'r', encoding='utf-8'))
    cur_rows = []
    append = cur_rows.append
    token_length = 0
    for line in tqdm(data, total=len(data)):
        raw_line_cat += 1
        input = line["instruction"]
        output = line['output']
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
    print(f"文件{read_file}处理完成, 用时{end_time - start_time}秒, 原始数据{raw_line_cat}条, 保留数据{keep_line_cat}条, token长度{token_length}")

    


def process_wiki_simple(groups_cnt=10000, max_len=512):
    """
    将繁体转换为简体 作为训练tokenizer的数据
    """
    file_name = os.path.join(data_dir, 'zhwiki', 'wiki.txt')
    save_file_name = os.path.join(processed_file_dir, 'zhwiki_simple.txt')

    start_time = time.time()
    if os.path.exists(save_file_name):
        assert whether_deleta_file(save_file_name)
    cc = OpenCC('t2s')
    def process_line(line: str) ->str:
        line = cc.convert(line)
        line = re.sub(r"\「|\」|\｢|\｣|\『|\』", '\"', line)
        line = re.sub(r"\，\）|\；\）", '）', line)  # 罗德·法尼(Rod Dodji Fanni，）
        line = re.sub(r"\（\，|\(\，", '（', line)  # 阿魯拉·基馬(Alula Girma (，

        line = convert_en_punctuation_to_zh_punct(line)
        line = remove_duplicate_punctuation(line)
        return line 
    with open(file_name, 'r', encoding='utf-8') as read_file:
        with open(save_file_name, 'a', encoding='utf-8') as f:
            total_len = len(read_file.readlines())
            read_file.seek(0)
            for line in tqdm(read_file, total=total_len):
                line = process_line(line)
                f.write(line)
    f.close()
    end_time = time.time()
    print(f"文件{file_name}处理完成, 用时{end_time - start_time}秒")
def process_wiki(groups_cnt=10000, max_len=512):
    """
    作为pretrained的数据
    """
    file_name = os.path.join(data_dir, 'zhwiki', 'wiki.txt')
    save_file_name = os.path.join(processed_file_dir, 'zhwiki.parquet')

    start_time = time.time() 
    if os.path.exists(save_file_name):
        assert whether_deleta_file(save_file_name)
    
    cc = OpenCC('t2s')
    all_cnt, keep_cnt = 0, 0
    
    prompt_prefix = [
        '什么是{}？',
        '介绍一下{}',
        '介绍一下什么是{}',
        '写一篇关于{}的介绍',
        '{}是什么？',
        '你知道{}吗？',
        '生成关于{}的介绍',
        '我想知道关于{}的详细信息',
        '你了解{}吗？',
        '请解释一下{}',
        '对于{}，你有什么了解或看法吗？',
        '请告诉我关于{}的信息',
        '请简要描述一下{}',
        '请提供有关{}的一些详细信息',
        '能否解释一下{}是什么?',
        '请分享一些关于{}的背景知识',
        '请简要概括一下{}',
        '能给我一些关于{}的背景资料吗?',
        '有关{}的信息可以分享一下吗？',
        '你能告诉我{}是什么吗？',
    ]

    def procees_line(line: str) -> str:
        '''
        处理一行文本
        '''
        # 将繁体转换为简体
        line = cc.convert(line)

        line = re.sub(r"\「|\」|\｢|\｣|\『|\』", '\"', line)  # 将「」｢｣『』这些符号替换成引号
        line = re.sub(r"\，\）|\；\）", '）', line)  # 罗德·法尼(Rod Dodji Fanni，）
        line = re.sub(r"\（\，|\(\，", '（', line)  # 阿魯拉·基馬(Alula Girma (，
        
        line = convert_en_punctuation_to_zh_punct(line) # 英文标点转换为中文标点
        line = remove_duplicate_punctuation(line)  # 删除中文空括号和重复的标点

        return line
    np.random.seed(0)
    choice = np.random.choice
    token_length = 0
    with open(file_name, 'r', encoding='utf-8') as read_file:
        prompt = '' 
        response = '' 
        pre_line_len = 0
        cur_rows = []
        append = cur_rows.append
        total_len = len(read_file.readlines())
        read_file.seek(0)
        for line in tqdm(read_file, total=total_len):
            all_cnt +=1
            if len(prompt) == 0 and pre_line_len > 0:
                pre_line_len = len(line.strip())
                continue
            line = procees_line(line)
            if prompt == '' and line.endswith('：') and pre_line_len == 0:
                prompt = choice(prompt_prefix).format(line[0: -1])
                continue
            pre_line_len = len(line.strip())
            if prompt != '' and not line.endswith('：'):
                # 其实，pre_line_len已经是len(line.strip())了，如果len(line.strip())=0，既是当前行是0，则不管答案长度够不够，都需要保存了
                if len(response) + len(line) <= max_len and pre_line_len != 0: 
                    response = '{}{}'.format(response, line)
                elif len(response) + len(line) > max_len or pre_line_len == 0:
                    # 长度超了或者当前的百科已经结束，保存一条样例
                    keep_cnt += 1
                    response = '{}{}'.format(response, line)
                    append({'prompt': prompt + response + '<eos>'})
                    token_length += len(prompt) + len(response)
                    prompt = ''
                    response = ''
            if len(cur_rows) >= groups_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(save_file_name, df)
                cur_rows = []
                append = cur_rows.append
        if len(prompt) > 0 and len(response) > 0:
            keep_cnt += 1
            append({'prompt': prompt + response + '<eos>'})
        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(save_file_name, df)
            cur_rows = []
        end_time = time.time()
    print(f"文件{file_name}处理完成, 用时{end_time - start_time}秒, 原始数据{all_cnt}条, 保留数据{keep_cnt}条, token长度{token_length}")

def process_belle_data():
    """
    处理belle数据集
    """
    belle_dir = os.path.join(data_dir, 'belle_pretrain')
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


def merge_dataset_as_single_file():
    """
    感觉不同的压缩差距蛮大的 使用from fastparquet import write 直接超过允许的最大值
    但是pandas的to_parquet可以
    而且压缩完之后 比几个文件加起来的大小还要小
    so good!
    """
    save_file_name = os.path.join(processed_file_dir, 'all_data.parquet')
    if os.path.exists(save_file_name):
        assert whether_deleta_file(save_file_name)
    
    parquet_files = [name  for name in os.listdir(processed_file_dir) if name.endswith('parquet') ]
    
    dataframes = [pd.read_parquet(os.path.join(processed_file_dir, file)) for file in parquet_files]
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    combined_df.to_parquet(save_file_name)


def get_pretrain_data(data_all_dir: str='LLama3/dataset/processed/all_data.parquet',save_path: str = 'LLama3/dataset/processed/pretrain_data.parquet'): 
    from transformers import PreTrainedTokenizerFast
    from datasets import Dataset
    tokenizer = PreTrainedTokenizerFast.from_pretrained('LLama3/tokenizer/fast_tokenizer')
    data_all = Dataset.from_parquet(data_all_dir)
    if os.path.exists(save_path):
        assert whether_deleta_file(save_path)
    
    def map_function(examples):
        input_ids = tokenizer(examples['prompt'], padding=False, truncation=False)['input_ids']
        return {'input_ids': input_ids}
    
    data_all = data_all.map(map_function, batched=True, remove_columns=['prompt'])

    token_ids_batch = []
    token_ids_all = []
    """
    形成batch
    """
    token_ids_length = 0
    for line in tqdm(data_all, total=len(data_all)):
        input_id = line['input_ids']
        token_ids_length += len(input_id)
        token_ids_batch.extend(input_id)
        while(len(token_ids_batch) > 512):
            token_ids_all.append({'input_ids': token_ids_batch[:512]})
            token_ids_batch = token_ids_batch[512:]
        if len(token_ids_all) >= 10000:
            token_ids_all_dataframe = pd.DataFrame(token_ids_all)
            write_single_parquet_file(save_path, token_ids_all_dataframe)
            token_ids_all= []
    while len(token_ids_batch) >= 512:
        token_ids_all.append({'input_ids': token_ids_batch[:512]})
        token_ids_batch = token_ids_batch[512:]
    if len(token_ids_all) > 0:
        token_ids_all_dataframe = pd.DataFrame(token_ids_all)
        write_single_parquet_file(save_path, token_ids_all_dataframe)
    print(f"处理完成, token长度{token_ids_length}")
if __name__ == "__main__":
    # # 处理shareAI的数据
    # process_shareAI_data()

    # # 处理弱智吧的数据
    # process_ruozhiba_data()

    # # 处理zhwiki的数据
    # process_wiki()
    
    # # 处理belle的数据
    # process_belle_data()

    # # 合并数据集
    # merge_dataset_as_single_file()

    # 使用zhwiki的简体数据训练完tokenizer之后 生成最终的pretrain数据集
    # 本来想放到train的数据预处理中，但发现数据集太大 太消耗内存了 

    get_pretrain_data()
    
    
    
