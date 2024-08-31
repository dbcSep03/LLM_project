# coding=utf-8
import json
import sys
import re
import numpy as np
from text2vec import SentenceModel, semantic_search, Similarity


simModel_path = './pre_train_model/text2vec-base-chinese'  # 相似度模型路径
simModel = SentenceModel(model_name_or_path=simModel_path, device='cuda:0')

def report_score(gold_path, predict_path):
    gold_info = json.load(open(gold_path))
    pred_info = json.load(open(predict_path))

    idx = 0
    for gold, pred in zip(gold_info, pred_info):
        question = gold["question"]
        gold = gold["answer"].strip()
        pred = pred["answer"].strip()
        if gold == "无答案" and pred != gold:
            score = 0.0
        else:
            score = semantic_search(simModel.encode([gold]), simModel.encode(pred), top_k=1)[0][0]['score']
        gold_info[idx]["score"] = score
        gold_info[idx]["predict"] = pred 
        idx += 1
        print(f"预测: {question}, 得分: {score}")

    return gold_info


if __name__ == "__main__":
    '''
      online evaluation
    '''

    # 标准答案路径
    gold_path = "./data/gold.json" 
    print("Read gold from %s" % gold_path)

    # 预测文件路径
    predict_path = "data/result_1.json" 
    print("Read predict file from %s" % predict_path)

    results = report_score(gold_path, predict_path)

    # 输出最终得分
    final_score = np.mean([item["score"] for item in results])
    print("\n")
    print("="*100)
    print(f"预测问题数：{len(results)}, 预测最终得分：{final_score}")
    print("="*100)

    # 结果文件路径
    metric_path = "./data/metrics.json" 
    results_info = json.dumps(results, ensure_ascii=False, indent=2)
    with open(metric_path, "w") as fd:
        fd.write(results_info)
    print(f"\n结果文件保存至{metric_path}")

