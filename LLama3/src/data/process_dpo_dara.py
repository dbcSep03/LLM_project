import pandas as pd

def process_huozi_rlhf_data(data_dir):
    data = pd.read_json(data_dir)
    data = data.drop(columns=['id'])
    return data
def process_beyond_rlhf_data(data_dir):
    data = pd.read_parquet(data_dir)
    data["reject"] = data["rejected"]
    data = data.drop(columns=['rejected'])
    return data
if __name__ == "__main__":
    target_dir = "LLama3/dataset/processed/rlhf.parquet"
    huozi_data  = process_huozi_rlhf_data('LLama3/dataset/DPO/huozi_rlhf_data.json')
    beyond_data = process_beyond_rlhf_data('LLama3/dataset/DPO/rlhf-reward-single-round-trans_chinese.parquet')
    data_all = pd.concat([huozi_data, beyond_data],ignore_index=True)
    data_all.to_parquet(target_dir)

