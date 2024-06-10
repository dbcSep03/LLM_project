import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from src.models.model import LLamamodel
from accelerate import load_checkpoint_and_dispatch
from transformers import PreTrainedTokenizerFast
from src.models.config import modleConfig
def chat_bot():
    tokenizer = PreTrainedTokenizerFast.from_pretrained('LLama3/tokenizer/fast_tokenizer')
    config = modleConfig(vocab_size = len(tokenizer), padding_idx=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    
    model = LLamamodel(config)
    model = load_checkpoint_and_dispatch(model, 'LLama3/checkpoints/LLama_pretrain/model.safetensors', device_map='auto')
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