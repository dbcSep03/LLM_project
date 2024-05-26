from dataclasses import dataclass

@dataclass
class TokenizerConfig:
    vocab_size: int = 65535
    data_dir: str = 'T5_model/data/processed'
    low_tokenizer: str = 'T5_model/pre_tokenizer/low_tokenizer'
    fast_tokenizer: str = 'T5_model/pre_tokenizer/fast_tokenizer'

@dataclass
class TrainConfig:
    data: str = 'T5_model/data/processed/all_data.parquet'
    tokenizer_fast: str = 'T5_model/pre_tokenizer/fast_tokenizer'
    mixed_precision: str = 'bf16'
    max_seq_length: int = 256
    output_dir: str = 'T5_model/pretrain'
    learning_rate: float = 5e-5
    max_epochs: int = 8
    gradient_accumulation_steps: int = 8
    learn_rate: float = 0.0001                      
    div_factor: int = 50
    warmup_steps: int = 1024
    batch_size_per_gpu: int = 8
    batch_size: int = 12
    output_dir_pytorch: str = 'T5_model/pytorch_model'


@dataclass
class ModelConfig:
    d_ff: int = 2048

    d_model: int = 768
    num_heads: int = 12
    d_kv: int = 64

    num_layers: int = 6
    num_decoder_layers: int = 6


@dataclass
class SFTConfig:
    max_seq_length: int = 256
    train_data: str = 'dataset/sft.parquet'
