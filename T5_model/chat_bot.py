import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import T5ForConditionalGeneration,PreTrainedTokenizerFast

def chat_bot():
    model = T5ForConditionalGeneration.from_pretrained('T5_model/dpo_model/checkpoint-1242')
    tokenizer = PreTrainedTokenizerFast.from_pretrained('T5_model/pre_tokenizer/fast_tokenizer')
    text = None
    model.cuda()
    while(True):
        text = input('You: ')
        if 'quit' in text.lower():
            break
        input_ids = tokenizer(f'{text}[EOS]', return_tensors='pt').input_ids
        output = model.generate(input_ids.cuda(), max_length=1000, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Bot: {response}')

if __name__ == '__main__':
    chat_bot()