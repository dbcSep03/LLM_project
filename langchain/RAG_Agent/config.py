import os

import torch

# device config
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
LLM_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
num_gpus = torch.cuda.device_count()

# model cache config
MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'model_cache')


# vector storage config
VECTOR_STORE_PATH='./vector_store'
COLLECTION_NAME='my_collection'
