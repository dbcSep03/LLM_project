from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import datasets
from itertools import chain
from tqdm.auto import tqdm
import torch
import json
import numpy as np
import re
from collections import defaultdict
def token_map(exapmle):
    input = exapmle['input']
    output = exapmle['target']
    input = "<|im_start|>问：\n{}\n答：\n".format(input.strip().replace('答：', '').strip())
    output = "{}\n<|im_end|>".format(output.strip())
    input_len = len(tokenizer(input, add_special_tokens=False)['input_ids'])

    results = tokenizer(text=input+output, add_special_tokens=False)
    results["labels"] = [-100]*input_len + results["input_ids"][input_len:]
    return {'input_ids': results['input_ids'], 'labels': results['labels']}
def get_entity(text, pattern, entity_types):
    result_list = []
    matches = pattern.findall(text)
    entity_dict = {etype: [] for etype in entity_types}

    for entity_type, entity in matches:
        entity_type = entity_type.replace("实体", "")
        if entity_type in entity_dict:
            entity_dict[entity_type].append(entity)

    for etype, entities in entity_dict.items():
        if entities:
            result_list.append({etype: entities})
    return result_list

def calculate_f1(answer, pred):
    true_positive = len(set(answer) & set(pred))
    precision = true_positive / len(pred) if pred else 0
    recall = true_positive / len(answer) if answer else 0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_f1_scores(data):
    f1_scores = defaultdict(list)
    
    for item in data:

        answer_dict = defaultdict(list)
        pred_dict = defaultdict(list)
        
        for ans in item['answer']:
            for key, value in ans.items():
                answer_dict[key].extend(value)
        
        for prd in item['pred']:
            for key, value in prd.items():
                pred_dict[key].extend(value)

        for entity_type in set(answer_dict.keys()).union(set(pred_dict.keys())):
            answer_entities = answer_dict[entity_type]
            pred_entities = pred_dict[entity_type]
            f1 = calculate_f1(answer_entities, pred_entities)
            f1_scores[entity_type].append(f1)

    average_f1_scores = {etype: sum(scores) / len(scores) for etype, scores in f1_scores.items()}
    
    return average_f1_scores


def get_results():
    model.eval()
    results_all = []
    entity_types = [
    "临床表现", "中医治则", "方剂", "西医诊断", "其他治疗",
    "西医治疗", "中医证候", "中药", "中医治疗", "中医诊断"]
    pattern = re.compile(r"(\w+实体)：([\u4e00-\u9fa5]+)")
    for batch in tqdm(test_data, total=len(test_data)):
        with torch.no_grad():
            input_ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()
            outputs = model(input_ids=input_ids, labels=labels)
            preds = outputs.logits.argmax(-1)
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            results_all.append(
                {"query": labels[0].split("答：")[0].strip(), 
                 "answer": get_entity(labels[0].split("答：")[1].strip(), pattern, entity_types), 
                 "pred": get_entity(decoded_preds[0].split("答：")[1].strip(), pattern, entity_types)

                }
            )
    with open(f'ruslts.json', 'w', encoding='utf-8') as json_file:
        json.dump(results_all, json_file, ensure_ascii=False, indent=4)
    f1_score = calculate_f1_scores(results_all)
    print(f1_score)

        


if __name__ == "__main__":
    
    model = AutoModelForCausalLM.from_pretrained("pretrain_model", trust_remote_code=True).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained("pretrain_model", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.decode(tokenizer.im_end_id)
    test_data = datasets.Dataset.from_json('data/test.json')
    test_data = test_data.map(token_map,remove_columns=["input", "target","answer_choices","task_dataset",'task_type', "sample_id"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    test_data = DataLoader(test_data, batch_size=1, collate_fn=data_collator)
    get_results()