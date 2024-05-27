from datasets import Dataset
import numpy as np
from transformers import T5Config
import torch
from typing import Union
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu

def get_data(data_dir,tokenizer, max_length):
    dataset = Dataset.from_parquet(data_dir)
    def process_example(example):
        prompt = example['prompt']
        response = example['response']
        prompt = tokenizer(prompt, padding=False, truncation=True, return_attention_mask=False, max_length=max_length)
        response = tokenizer(response, padding=False, truncation=True, return_attention_mask=False, max_length=max_length)
        prompt = [np.array(sent + [tokenizer.eos_token_id],dtype=np.uint16)for sent in prompt["input_ids"]]
        response = [np.array(sent + [tokenizer.eos_token_id], dtype=np.uint16)for sent in response["input_ids"]]
        return {"input_ids": prompt, "labels": response}
    return dataset.map(process_example, batched=True, remove_columns=dataset.column_names)

def get_data_for_dpo(data_dir, tokenizer, max_length):
    dataset = Dataset.from_parquet(data_dir)
    def process_example(example):
        prompts = [f"{prompt}[EOS]" for prompt in example['prompt']]
        chosens = [f"{chosen}[EOS]" for chosen in example['chosen']]
        rejecteds = [f"{rejected}[EOS]" for rejected in example['rejected']]
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}
    return dataset.map(process_example, batched=True, remove_columns=dataset.column_names)

def get_model_config(vocab_size, model_config, decoder_start_token_id, eos_token_id):
    config = T5Config()
    config.vocab_size = vocab_size
    config.d_model = model_config.d_model
    config.num_heads = model_config.num_heads
    config.d_kv = model_config.d_kv
    config.num_layers = model_config.num_layers
    config.num_decoder_layers = model_config.num_decoder_layers
    config.d_ff = model_config.d_ff
    config.decoder_start_token_id = decoder_start_token_id
    config.eos_token_id = eos_token_id
    return config


def extract_ngram(words_list: list[str], n_gram: int):
    '''
    获取一个句子的n_grama
    return：
        ngram_counter： key = ('w1  w2 ... wn', n_gram), value: count of key
    '''
    n = len(words_list)
    ngram_counter = Counter()

    for i in range(1, n_gram + 1):
        for j in range(n - i + 1):
            key = ' '.join(words_list[j: j + i])
            ngram_counter[(key, i)] += 1

    return ngram_counter


def get_bleu4_score(reference: Union[str, list[str]], outputs: Union[str, list[str]], n_gram: int = 4) -> float:
    '''
    获取bleu4分数
    '''

    weights = np.ones(n_gram) * (1.0 / n_gram)

    outputs_len, reference_len = len(outputs), len(reference)

    if not type(reference) is list:
        reference = list(reference)
    if not type(outputs) is list:
        outputs = list(outputs)

    outputs_counter = extract_ngram(outputs, n_gram=n_gram)
    reference_counter = extract_ngram(reference, n_gram=n_gram)

    ngram_counter_clip = outputs_counter & reference_counter

    clip_counter = np.zeros(n_gram)
    output_ngram_counter = np.zeros(n_gram)

    for (key, ngram), cnt in ngram_counter_clip.items():
        clip_counter[ngram - 1] += cnt

    for (key, ngram), cnt in outputs_counter.items():
        output_ngram_counter[ngram - 1] += cnt

    if np.min(clip_counter) == 0.0:
        return np.array(0.0)

    precision_scores = clip_counter / output_ngram_counter

    # bleu
    log_precision_scores = weights * np.log(precision_scores)

    # 几何平均形式求平均值然后加权
    geometric_mean = np.exp(np.sum(log_precision_scores))
    brevity_penalty = np.exp(1 - (reference_len / outputs_len))

    bleu = brevity_penalty * geometric_mean

    return bleu

if __name__ == "__main__":
    # ref = ['抱歉，我不我知道ABB代表什么意思', '天气好，你好啊']
    # out = ['我不明白ABB是什么意思', '好天气，我不好']
    ref = '我不我知道ABB代表什么意思'
    out = '我不明白ABB是什么意思'
    # b1 = sentence_bleu([list(out)], list(ref), weights=(0.25, 0.25, 0.25, 0.25))
    # print(b1)
    b2 = sentence_bleu(out, ref)
    print(b2)
    print('----')
    candidate_corpus = ['i', 'have', 'a', 'pen', 'on', 'my', 'desk', 'a', 'b', 'c', 'd', 'f', 'f']
    reference_corpus = ['there', 'is', 'a', 'pen', 'on', 'my', 'desk', 'a', 'b', 'd', 'd', 'fd']

    # print(sentence_bleu([reference_corpus], candidate_corpus, weights=(0.25, 0.25, 0.25, 0.25)))
    print(get_bleu4_score(reference_corpus, candidate_corpus))
