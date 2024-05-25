import tokenizers
from tokenizers import Tokenizer, decoders
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Metaspace, Digits, Punctuation
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from config import TokenizerConfig
from transformers import PreTrainedTokenizerFast
import os
def main():
    special_tokens = ["[PAD]", "[EOS]", "[SEP]", "[BOS]", "[CLS]", "[MASK]", "[UNK]"]
    model = BPE(unk_token="[UNK]")
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = tokenizers.normalizers.Sequence([NFKC()])
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([Punctuation(), Digits(individual_digits=True), Metaspace()])
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.decoder = decoders.Metaspace()

    trainer = BpeTrainer(vocab_size=TokenizerConfig.vocab_size, min_frequency=100, show_progress=True, special_tokens=special_tokens)
    corpus = [os.path.join(TokenizerConfig.data_dir, name) for name in os.listdir(TokenizerConfig.data_dir) if name.endswith(".txt")]
    
    # tokenizer.train(corpus, trainer=trainer) 无法一次训练完 

    def get_trianing_corpus(buffer_size: int=1000, chunk_len=2048)->list:
        """
        一个文本块2048个字符
        """
        line_cnt = 0
        buffer = []
        for file_name in corpus:
            with open(file_name, "r", encoding="utf-8") as f:
                cur_chunk_text, txt_len = [], 0
                for line in f:
                    line_cnt += 1
                    txt_len += len(line)
                    cur_chunk_text.append(line)
                    if txt_len >= chunk_len:
                        buffer.append(''.join(cur_chunk_text))
                        cur_chunk_text, txt_len = [], 0
                    if len(buffer) >= buffer_size:
                        yield buffer
                        buffer = []
        if len(buffer) > 0:
            yield buffer
    tokenizer.train_from_iterator(get_trianing_corpus(), trainer=trainer)  # 对文本进行迭代，对每次的buffer进行训练

    
    if '\t' not in tokenizer.get_vocab():
        tokenizer.add_tokens(["\t"])
    if '\n' not in tokenizer.get_vocab():
        tokenizer.add_tokens(["\n"])
    
    tokenizer.save(TokenizerConfig.low_tokenizer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    fast_tokenizer.save_pretrained(TokenizerConfig.fast_tokenizer)
if __name__ == "__main__":
    main()
