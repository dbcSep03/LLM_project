from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Digits, Punctuation, Metaspace
from tokenizers.normalizers import NFKC
import tokenizers
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
def train_tokenizer():
    special_tokens = ['<EOS>', '<BOS>', '<SEP>', '<CLS>', '<MASK>', '<UNK>','<PAD>']
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([Punctuation(), Digits(individual_digits=True), Metaspace()])
    tokenizer.normalizer = NFKC()
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.decoder = tokenizers.decoders.Metaspace()
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=65535, min_frequency=50)
    tokenizer.train(files=['LLama3/dataset/processed/zhwiki_simple.txt'], trainer=trainer)
    tokenizer.save('LLama3/tokenizer/slow_tokenizer')
    if '\t' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['\t'])
    if '\n' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['\n'])
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token='<UNK>',
        pad_token='<PAD>',
        cls_token='<CLS>',
        sep_token='<SEP>',
        eos_token='<EOS>',
        bos_token='<BOS>',
        mask_token='<MASK>',
        )
    fast_tokenizer.save_pretrained('LLama3/tokenizer/fast_tokenizer')
    
    
if __name__ == '__main__':
    train_tokenizer()