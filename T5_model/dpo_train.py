import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from trl import DPOTrainer
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration, TrainingArguments
from config import DPOConfig
from utils.funcation import get_data_for_dpo

def DpoTrain():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(DPOConfig.tokenizer_fast)
    model = T5ForConditionalGeneration.from_pretrained(DPOConfig.model_path)
    ref_model = T5ForConditionalGeneration.from_pretrained(DPOConfig.model_path)
    data = get_data_for_dpo(DPOConfig.dpo_data, tokenizer, DPOConfig.max_length)
    train_args = TrainingArguments(
        output_dir=DPOConfig.output_dir,
        auto_find_batch_size=True,
        overwrite_output_dir=True,
        do_eval=False,
        per_device_train_batch_size=DPOConfig.batch_size_per_gpu,
        per_device_eval_batch_size=DPOConfig.batch_size_per_gpu,
        gradient_accumulation_steps=DPOConfig.gradient_accumulation_steps,
        learning_rate=DPOConfig.learning_rate,
        num_train_epochs=DPOConfig.max_epochs,
        log_level='info',
        report_to='wandb',
        save_strategy='epoch',
        save_total_limit=3,
        bf16=True,
    )
    dpo_train = DPOTrainer(
        model = model, 
        ref_model=ref_model, 
        train_dataset=data, 
        tokenizer=tokenizer, 
        args=train_args,
        max_length=DPOConfig.max_length,
        max_target_length=DPOConfig.max_length,
        max_prompt_length=DPOConfig.max_length,
        generate_during_eval=True,
        is_encoder_decoder=True,
        )
    dpo_train.train()


if __name__ == '__main__':
    DpoTrain()