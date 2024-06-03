import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from src.models.model import LLamamodel
from accelerate import load_checkpoint_and_dispatch
from transformers import PreTrainedTokenizerFast

def chat_bot():
    model = LLamamodel()
    model = load_checkpoint_and_dispatch(model, 'LLama3/checkpoints/LLama_pretrain.pth/model.safetensors', map_location='auto')
    tokenizer = PreTrainedTokenizerFast.from_pretrained('LLama3/tokenizer/fast_tokenizer')
    while(True):
        text = input("input:")
        if text == 'exit':
            break
        input_id = tokenizer(text, return_tensors='pt')['input_ids']
        output = model.generate(input_id)
        print(tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == '__main__':
    chat_bot()