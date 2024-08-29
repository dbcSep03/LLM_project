import pandas as pd
import json
import random
def get_label(texts, labels,sample_id, label_choice, input, label2id):
    
    results = {}
    results["input"] = input
    results["task_type"] = 'ner'
    results['task_dataset'] = 'custom_data'
    results['sample_id'] = sample_id
    results['answer_choices'] = list(label_choice)
    # 获取 id 到标签的反向映射
    id2label = {v: k for k, v in label2id.items()}

    # 存储最终提取的结果
    extracted_spans = []

    # 临时存储当前标签和对应文本
    current_label = None
    current_text = []

    for char, label_id in zip(texts, labels):
        label = id2label[label_id]
        if label != 'O':
            # 如果遇到新标签或者不同的 "B-" 开头的标签，保存之前的结果
            if current_label is None or label.startswith('B-') or (label != current_label and not label.startswith('I-')):
                if current_text:
                    extracted_spans.append((current_label, ''.join(current_text)))
                    current_text = []
                current_label = label
            
            # 将字符添加到当前文本中
            current_text.append(char)
        else:
            # 当遇到 'O' 标签时，保存之前的结果并重置
            if current_text:
                extracted_spans.append((current_label[2:], ''.join(current_text)))
                current_text = []
            current_label = None

    # 处理最后的文本块
    if current_text:
        extracted_spans.append((current_label[2:], ''.join(current_text)))

    # 打印提取结果
    output = "上述句子中的实体包含：\n"
    answer = []
    for label, text_span in extracted_spans:
        answer.append(f"{label}实体：{text_span}")
    output += '\n'.join(answer)
    results['target'] = output
    return results

def read_data(file_path):

    # 初始化变量
    texts = []
    labels = []
    current_text = []
    current_labels = []

    # 打开文件并读取内容
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # 去除行尾的换行符和多余的空格
            
            if line == '':
                # 如果遇到空行，代表一个文本段落结束
                if current_text and current_labels:
                    texts.append(''.join(current_text))
                    labels.append(' '.join(current_labels))
                    current_text = []
                    current_labels = []
            else:
                # 非空行，继续收集文本和标签
                parts = line.split()
                if len(parts) == 2:
                    word, label = parts
                    current_text.append(word)
                    current_labels.append(label)
        
        # 处理文件末尾的最后一段（如果没有空行结尾）
        if current_text and current_labels:
            texts.append(''.join(current_text))
            labels.append(' '.join(current_labels))
    
    labels_id = set()
    for label in labels:
        for l in label.split():
            labels_id.add(l)
    label2id = {label: idx for idx, label in enumerate(labels_id)}
    id2label = {idx: label for label, idx in label2id.items()}
    label_all_id = []
    for label in labels:
        label_all_id.append([label2id[l] for l in label.split(' ')])
    label_choice = set()
    for a in label2id.keys():
        if len(a) > 1:
            label_choice.add(a[2:])
    return texts, label_all_id, label2id, list(label_choice)

def create_data(texts, label_all_id, label_choice, label2id, output_name):
    template = [
        "找出指定的实体：\\n[INPUT_TEXT]\\n类型选项：[LIST_LABELS]\\n答：",
        "找出指定的实体：\\n[INPUT_TEXT]\\n实体类型选项：[LIST_LABELS]\\n答：",
        "找出句子中的[LIST_LABELS]实体：\\n[INPUT_TEXT]\\n答：",
        "[INPUT_TEXT]\\n问题：句子中的[LIST_LABELS]实体是什么？\\n答：",
        "生成句子中的[LIST_LABELS]实体：\\n[INPUT_TEXT]\\n答：",
        "下面句子中的[LIST_LABELS]实体有哪些？\\n[INPUT_TEXT]\\n答：",
        "实体抽取：\\n[INPUT_TEXT]\\n选项：[LIST_LABELS]\\n答：",
        "医学实体识别：\\n[INPUT_TEXT]\\n实体选项：[LIST_LABELS]\\n答："
    ]
    results_all = []
    for i, (text, label) in enumerate(zip(texts, label_all_id)):
        index = random.randint(0,7)
        input = template[index].replace('[INPUT_TEXT]', text).replace('[LIST_LABELS]', '，'.join(list(label_choice)))
        results = get_label(text, label, f'f{output_name}_{i}', label_choice, input, label2id)
        results_all.append(results)
    with open(f'data/{output_name}.json', 'w', encoding='utf-8') as json_file:
        json.dump(results_all, json_file, ensure_ascii=False, indent=4)
    
if __name__ == '__main__':
    # for data_str in ['medical.dev', 'medical.train', 'medical.test']:
    for data_str in ['medical.train']:
        texts, label_all_id, label2id, label_choice = read_data(f'data/{data_str}')
        create_data(texts, label_all_id, label_choice, label2id, data_str[8:])

