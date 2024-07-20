from transformers import PreTrainedTokenizerFast

if __name__ == '__main__':
    tokenizer = PreTrainedTokenizerFast.from_pretrained('LLama3/tokenizer/fast_tokenizer')
    print(tokenizer.eos_token)
    print(tokenizer('<eos>'))