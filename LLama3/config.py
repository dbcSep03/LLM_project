from dataclasses import dataclass
@dataclass
class train_config:
    embedding_size: int
    hidden_size: int
    layer: int
    num_heads: int
    num_key_value_heads: int
    bias: bool
    seq_length: int
    rotary_base: int = 10000
    intermediate_size: int
