import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, GenerationConfig
from utils.funcation import get_data, get_model_config
from config import TrainConfig, ModelConfig

def train():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TrainConfig.tokenizer_fast)
    model_config = get_model_config(vocab_size=len(tokenizer.get_vocab()) , model_config=ModelConfig, decoder_start_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    model = T5ForConditionalGeneration(model_config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_params = sum(p.numel() for p in model.shared.parameters())
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}, embedding_params: {embedding_params}, model_size: {total_params/1e9} B")

    data = get_data(TrainConfig.data, tokenizer)

    data = data.train_test_split(test_size=0.1)
    
    generation_config = GenerationConfig()
    generation_config.remove_invalid_values = True
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.decoder_start_token_id = tokenizer.pad_token_id
    generation_config.max_length = 320
    generation_config.num_beams = 1
    generation_config.do_sample = False

    collator = DataCollatorForSeq2Seq(tokenizer)
    
    trainargs = Seq2SeqTrainingArguments(
        output_dir=TrainConfig.output_dir,
        per_device_train_batch_size=TrainConfig.batch_size_per_gpu,
        per_device_eval_batch_size=TrainConfig.batch_size_per_gpu,
        auto_find_batch_size=True,
        gradient_accumulation_steps=TrainConfig.gradient_accumulation_steps,
        learning_rate=TrainConfig.learning_rate,
        num_train_epochs=TrainConfig.max_epochs,
        save_strategy='epoch',
        eval_strategy='epoch',
        optim="adafactor",
        save_total_limit=3,
        prediction_loss_only=True,
        logging_steps=100,
        logging_dir=TrainConfig.output_dir,
        report_to='wandb',
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        bf16=True,
        generation_config=generation_config
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=trainargs,
        data_collator=collator,
        train_dataset=data['train'],
        eval_dataset=data['test'],
        tokenizer=tokenizer,
    )
    trainer.train()

if __name__ == '__main__':
    train()