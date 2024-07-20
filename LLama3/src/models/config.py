from dataclasses import dataclass
@dataclass
class modleConfig:
    vocab_size: int
    padding_idx: int
    eos_token_id: int
    hidden_size: int = 1024
    layer: int = 10
    num_heads: int = 16
    num_key_value_heads: int = 8
    bias: bool=False
    seq_length: int = 512
    rotary_base: int = 10000
    intermediate_size: int=2048
    dropout: float=0
    eps: float=1e-5


@dataclass
class trainConfig:
    tokenizer_path: str='LLama3/tokenizer/fast_tokenizer'
    dataset_path: str='LLama3/dataset/processed/pretrain_data.parquet'
    pretrain_data_path: str='LLama3/dataset/processed/pretrain_data.parquet'
    seq_length: int = 512
    epochs: int = 8
    gradient_accumulation_steps: int = 8
    batch_size: int = 8

@dataclass
class SFTConfig:
    tokenizer_path: str='LLama3/tokenizer/fast_tokenizer'
    dataset_path: str='LLama3/dataset/sft/sft_data_token.parquet'
    model_path: str='LLama3/checkpoints/LLama_pretrain/model.safetensors'
    seq_length: int = 512
    epochs: int = 8
    gradient_accumulation_steps: int = 8
    batch_size: int = 8

