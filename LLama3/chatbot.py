import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from src.models.model import LLamamodel
from accelerate import load_checkpoint_and_dispatch
from src.models.config import loraConfig
from src.models.lora import LoRA_model
from transformers import PreTrainedTokenizerFast
from src.models.config import modleConfig
import torch
def chat_bot():
    tokenizer = PreTrainedTokenizerFast.from_pretrained('LLama3/tokenizer/fast_tokenizer')
    config = modleConfig(vocab_size = len(tokenizer), padding_idx=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    
    model = LLamamodel(config)
    # is_lora = True
    # if is_lora:
    #     lora_config = loraConfig()
    #     model = LoRA_model(model, lora_config.r, lora_config.alpha, lora_config.lora_name, tokenizer.eos_token_id)
    #     model = load_checkpoint_and_dispatch(model, 'LLama3/checkpoints/LLama_sft_lora/model.safetensors', device_map='auto')
    # else:
    #     model = load_checkpoint_and_dispatch(model, 'LLama3/checkpoints/LLama_sft/model.safetensors', device_map='auto')
    model.load_state_dict(torch.load('LLama3/checkpoints/LLama_rlhf/model.pth'))
    model.to('cuda')
    while(True):
        text = input("input:")
        if text == 'exit':
            break
        input_id = tokenizer(text, return_tensors='pt')['input_ids'].cuda()
        output = model.generate(input_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Bot: {response[len(text)+1:]}')

if __name__ == '__main__':
    chat_bot()