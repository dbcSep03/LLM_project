from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "weight/Qwen_7B_chat"
quant_path = "weight/Qwen_7B_chat_int4_AWQ"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True}, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model.quantize(tokenizer, quant_config=quant_config,calib_data="pile-val-backup",split="validation")
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)