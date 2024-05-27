import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer,GenerationConfig
from config import SFTConfig
from utils.funcation import get_data
def sft_train():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(SFTConfig.fast_tokenizer)
    model = T5ForConditionalGeneration.from_pretrained(SFTConfig.sft_model)
    # 冻结了embedding层
    model.shared.requires_grad = False
    generation_config = GenerationConfig()
    generation_config.remove_invalid_values = True
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.decoder_start_token_id = tokenizer.pad_token_id
    generation_config.max_new_tokens = 320
    generation_config.repetition_penalty = 1.5
    generation_config.num_beams = 1         # greedy search
    generation_config.do_sample = False     # greedy search
    data = get_data(SFTConfig.train_data, tokenizer, 255)
    data = data.train_test_split(test_size=0.1)
    datacollactor = DataCollatorForSeq2Seq(tokenizer)
    train_args = Seq2SeqTrainingArguments(
        output_dir='T5_model/sft_model',
        per_device_train_batch_size=SFTConfig.batch_size_per_gpu,
        per_device_eval_batch_size=SFTConfig.batch_size_per_gpu,
        auto_find_batch_size=True,
        gradient_accumulation_steps=SFTConfig.gradient_accumulation_steps,
        learning_rate=SFTConfig.learning_rate,
        num_train_epochs=SFTConfig.max_epochs,
        save_strategy='epoch',
        eval_strategy='epoch',
        optim='adafactor',
        save_total_limit=3,
        load_best_model_at_end=True,
        bf16=True,
        overwrite_output_dir=True,
        do_eval=True,
        do_train=True,
        logging_steps=100,
        # report_to='wandb',
        log_level='info',
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        data_collator=datacollactor,
        train_dataset=data['train'],
        eval_dataset=data['test'],
    )
    trainer.train()

if __name__ == '__main__':
    sft_train()