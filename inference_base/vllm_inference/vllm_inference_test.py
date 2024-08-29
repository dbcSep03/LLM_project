from vllm import LLM, SamplingParams
import time
import torch
def vllm_inference(str):
    llm = LLM(model="inference_base/vllm_inference/weights/qwen_7b", trust_remote_code=True)
    print("单卡无量化")
    for i in [1,4,8,16,32]:
        prompts = [str for _ in range(i)]
        start_time = time.time()
        results = llm.generate(prompts, sampling_params=sampling_params)
        keep_time = time.time() - start_time
        print(f"{len(results)}: {keep_time} s")
    del llm 
    torch.cuda.empty_cache()
def vllm_int4_inference(str):
    llm = LLM(model="inference_base/vllm_inference/weights/qwen_7b_awq", quantization="AWQ",trust_remote_code=True,dtype="half")
    print("int4AWQ")
    for i in [1,4,8,16,32]:
        prompts = [str for _ in range(i)]
        start_time = time.time()
        results = llm.generate(prompts, sampling_params=sampling_params)
        keep_time = time.time() - start_time
        print(f"{len(results)}: {keep_time} s")
    del llm 
    torch.cuda.empty_cache()

def vllm_int8_kv_cache(str):
    llm = LLM(model="inference_base/vllm_inference/weights/qwen_7b", kv_cache_dtype="fp8", trust_remote_code=True)
    print("fp8kvcache")
    for i in [1,4,8,16,32]:
        prompts = [str for _ in range(i)]
        start_time = time.time()
        results = llm.generate(prompts, sampling_params=sampling_params)
        keep_time = time.time() - start_time
        print(f"{len(results)}: {keep_time} s")
    del llm 
    torch.cuda.empty_cache()

def vllm_parall_inference(str):
    
    llm = LLM(model="inference_base/vllm_inference/weights/qwen_7b", trust_remote_code=True, tensor_parallel_size=2)
    print("双卡并行")
    for i in [1,4,8,16,32]:
        prompts = [str for _ in range(i)]
        start_time = time.time()
        results = llm.generate(prompts, sampling_params=sampling_params)
        keep_time = time.time() - start_time
        print(f"{len(results)}: {keep_time} s")
    del llm 
    torch.cuda.empty_cache()

if __name__ == "__main__":
    prompt = "详细介绍一下北京的风土人情:"
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # vllm_inference(prompt)
    # vllm_int4_inference(prompt)
    # vllm_int8_kv_cache(prompt)
    vllm_parall_inference(prompt)
