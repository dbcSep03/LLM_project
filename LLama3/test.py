from datasets import Dataset
from torch.utils.data import DataLoader
from src.models.LLamadataset import LLamaDataset
from src.models.config import modleConfig, trainConfig
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
if __name__ == '__main__':
    data = Dataset.from_parquet('LLama3/dataset/processed/pretrain_data.parquet')
    data = data.select(range(10))
    tokenizer = PreTrainedTokenizerFast.from_pretrained('LLama3/tokenizer/fast_tokenizer')
    dataset = LLamaDataset(data, trainConfig, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    for index, batch in enumerate(dataloader):
        if index>0:
            break
        print(batch['input_ids'])
        print(batch['input_ids'].shape)
        print(tokenizer.batch_decode(batch['input_ids']))
        