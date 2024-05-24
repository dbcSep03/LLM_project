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
import pyarrow.parquet as pq

punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：\n") # 标点符号
data_dir = '/home/dongbingcheng/LLM_study/T5_model/data/raw'
processed_file_dir = '/home/dongbingcheng/LLM_study/T5_model/data/processed'
en_punctuation = ",().!;:"
zh_punctuation = "，（）。！；："

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
        choice = input("是否删除[y/n]")
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

def write_single_parquet_file(file_name: str, data: dict) ->None:
    """
    将数据写入parquet文件
    """
    append = False
    if os.path.exists(file_name):
        append = True
    write(file_name, data, compression='GZIP', append=append)


def process_line_data(title, desc, response, response_less_word: int=15) ->dict:
    """
    处理的流程
    先检测描述和标题是否相似
    如果相似则只用标题作为prompt 否则为标题+描述
    然后删除'\r'和重复的标点符号
    最后检测至少长度为15
    """
    if get_sentence_similarity(title, desc) > 0.9:
        prompt = title
    else:
        prompt = f"{title}{desc}"
    prompt = prompt.replace('\r', '')
    remove_duplicate_punctuation(prompt)
    response = response.replace('\r', '')
    remove_duplicate_punctuation(response)
    if len(response) < response_less_word or len(prompt) < response_less_word:
        return None
    else:
        return {'prompt': prompt, 'response': response}

def read_and_write_template(read_file: str, save_file_name: str, key_value: list[str], group_cnt: int=10000, special_f=None) ->None:
    """
    读取原始文件并写入处理后的文件
    """
    raw_line_cat = 0
    keep_line_cat = 0
    
    start_time = time.time()
    
    if read_file.endswith('json'):
        read = ujson.loads
        with open(read_file, 'r', encoding='utf-8') as f:
            cur_rows = []
            append = cur_rows.append

            total_len = len(f.readlines())
            f.seek(0)
            for line in tqdm(f, total=total_len):
                raw_line_cat += 1
                line = read(line)
                if special_f is not None:
                    if not special_f(line, key_value):
                        continue
                write_dict = process_line_data(line[key_value[0]], line[key_value[1]], line[key_value[2]])
                if write_dict is None: continue
                
                append(write_dict)
                keep_line_cat += 1

                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(save_file_name, df)
                    cur_rows = []
                    append = cur_rows.append
            
    if read_file.endswith('parquet'):
        data = pd.read_parquet(read_file)
        cur_rows = []
        append = cur_rows.append
        for index, row in tqdm(data.iterrows(),total=len(data)):
            raw_line_cat += 1
            if special_f is not None:
                if not special_f(row, key_value):
                    continue
            write_dict = process_line_data(row[key_value[0]], row[key_value[1]], row[key_value[2]])
            if write_dict is None: continue
            append(write_dict)
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
    print(f"文件{read_file}处理完成, 用时{end_time - start_time}秒, 原始数据{raw_line_cat}条, 保留数据{keep_line_cat}条")

def process_bake_qa(response_less_word: int=15) ->None:
    bake_qa_dir = os.path.join(data_dir, 'baike')
    file_names = os.listdir(bake_qa_dir)
    save_file_name = os.path.join(processed_file_dir, 'baike_qa.parquet')
    
    if os.path.exists(save_file_name):
        assert whether_deleta_file(save_file_name)

    for file in file_names:
        read_file = os.path.join(bake_qa_dir, file)
        read_and_write_template(read_file, save_file_name, key_value=["title", "desc", "answer"])


def process_belle():
    belle_dir = os.path.join(data_dir, 'belle_pretrain')
    file_names = os.listdir(belle_dir)
    save_file_name = os.path.join(processed_file_dir, 'belle.parquet')
    
    if os.path.exists(save_file_name):
        assert whether_deleta_file(save_file_name)

    def remove_data(line, key_value)->bool:
        title, desc, response = line[key_value[0]], line[key_value[1]], line[key_value[2]]
        # 删除翻译类任务
        if '翻译' in title or 'translate' in title.lower():
            return False
        
        # 删除表格类任务
        if '表格' in title or '-----' in title or '-----' in response:
            return False
        return True
    for file in file_names:
        read_file = os.path.join(belle_dir, file)
        read_and_write_template(read_file, save_file_name, key_value=["instruction","input", "output"], special_f=remove_data)

def process_webtext():
    webtext_dir = os.path.join(data_dir, 'webtext')
    file_names = os.listdir(webtext_dir)
    save_file_name = os.path.join(processed_file_dir, 'webtext.parquet')
    
    if os.path.exists(save_file_name):
        assert whether_deleta_file(save_file_name)
    def filter_by_star(line, key_value,keep_start=5):
        if line['star'] < keep_start:
            return False
        return True
    for file in file_names:
        read_file = os.path.join(webtext_dir, file)
        read_and_write_template(read_file, save_file_name, key_value=["title", "desc", "content"],special_f=filter_by_star)

def process_zhihu():
    zhihu_dir = os.path.join(data_dir, 'Zhihu-KOL')
    file_names = os.listdir(zhihu_dir)
    save_file_name = os.path.join(processed_file_dir, 'zhihu.parquet')

    if os.path.exists(save_file_name):
        assert whether_deleta_file(save_file_name)
    for file in file_names:
        read_file = os.path.join(zhihu_dir, file)
        read_and_write_template(read_file, save_file_name, key_value=['INSTRUCTION', 'INSTRUCTION', 'RESPONSE'])

def process_wiki(groups_cnt=10000, max_len=512):
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

    with open(file_name, 'r', encoding='utf-8') as read_file:
        prompt = '' 
        response = '' 
        pre_line_len = 0
        cur_rows = []
        append = cur_rows.append
        # total_len = len(read_file.readlines())
        # read_file.seek(0)
        total_len = 10914479
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
                    append({'prompt': prompt, 'response': ''.join(response[0: max_len])})
                    prompt = ''
                    response = ''
            if len(cur_rows) >= groups_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(save_file_name, df)
                cur_rows = []
                append = cur_rows.append
        if len(prompt) > 0 and len(response) > 0:
            keep_cnt += 1
            append({'prompt': prompt, 'response': response})
        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(save_file_name, df)
            cur_rows = []
        end_time = time.time()
    print(f"文件{file_name}处理完成, 用时{end_time - start_time}秒, 原始数据{all_cnt}条, 保留数据{keep_cnt}条")

def merge_dataset_as_single_file(groups_cnt: int=10000, min_len=3, cut_max_len: bool=False, max_len: int=512):
    save_file_name = os.path.join(processed_file_dir, 'all_data.parquet')
    if os.path.exists(save_file_name):
        assert whether_deleta_file(save_file_name)
    
    parquet_files = [name  for name in os.listdir(processed_file_dir) if name.endswith('parquet') ]
    
    cur_rows = []
    append = cur_rows.append
    
    all_cnt, keep_cnt = 0, 0
    for file in parquet_files:
        print(f"正在处理文件{file}")
        parquet_file = pq.read_table(os.path.join(processed_file_dir, file))
        for prompt, response in tqdm(zip(parquet_file['prompt'], parquet_file['response']), total=len(parquet_file)):
            prompt, response = prompt.as_py(), response.as_py()
            all_cnt += 1

            if len(prompt) < min_len or len(response) < min_len:
                continue
            if cut_max_len and (len(prompt) > max_len or len(response) > max_len):
                prompt = prompt[0: max_len]
                response = response[0: max_len]
            
            keep_cnt += 1
            append({'prompt': prompt, 'response': response})

            if len(cur_rows) >= groups_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(save_file_name, df)
                cur_rows = []
                append = cur_rows.append
    if len(cur_rows) >= groups_cnt:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_file_name, df)
        cur_rows = []
        append = cur_rows.append
    print(f"文件处理完成, 原始数据{all_cnt}条, 保留数据{keep_cnt}条")

def parquet_to_text(sep='[SEP]', buffer_size: int=50000) ->None:
    parquer_file = os.path.join(processed_file_dir, 'all_data.parquet')
    save_file_name = os.path.join(processed_file_dir, 'all_data.txt')

    if os.path.exists(save_file_name):
        assert whether_deleta_file(save_file_name)
    
    df = pd.read_parquet(parquer_file)

    cur_rows = []
    append = cur_rows.append
    with open(save_file_name, 'a', encoding='utf-8') as f:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            prompt, response = row['prompt'], row['response']
            append(f"{prompt}{sep}{response}\n")
            if len(cur_rows) >= buffer_size:
                f.writelines(cur_rows)
                cur_rows = []
                append = cur_rows.append
        if len(cur_rows) > 0:
            f.writelines(cur_rows)
            cur_rows = []

def parquet_to_json():
    save_file_name = os.path.join(processed_file_dir, 'all_data.json')
    parquer_file = os.path.join(processed_file_dir, 'all_data.parquet')

    if os.path.exists(save_file_name):
        assert whether_deleta_file(save_file_name)
    
    df = pd.read_parquet(parquer_file)

    cur_rows = []
    append = cur_rows.append
    with open(save_file_name, 'a', encoding='utf-8') as f:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            prompt, response = row['prompt'], row['response']
            append({"prompt": prompt, "response": response})
            
        with open(save_file_name, 'w', encoding='utf-8') as f:
            ujson.dump(cur_rows, f, indent=4, ensure_ascii=False)
if __name__ == "__main__":
    """
    原始数据在T5_model/data/raw
    处理后的数据在T5_model/data/processed
    """
    if not os.path.exists(processed_file_dir):
        os.makedirs(processed_file_dir)
    
    # 接下来依次处理数据集
    # 1. 处理baike数据集
    # process_bake_qa()

    # 2.处理belle数据集
    # process_belle()

    # 3.处理webtext数据集
    # process_webtext()

    # 4.处理zhihu数据集
    # process_zhihu()

    # 5.处理wiki数据集
    # process_wiki()

    # 将多个parquet合并为一个
    # merge_dataset_as_single_file()

    # 并没有执行minihash操作 进行去重 TODO

    # 保存为text
    # parquet_to_text()

    parquet_to_json()