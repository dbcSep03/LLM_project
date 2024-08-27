import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# os.environ["enforce_eager"] = "True"
import torch
import time
from typing import Optional, List, Any
from config import *
from vllm import LLM, SamplingParams

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids
from langchain_core.language_models import llms
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.pydantic_v1 import PrivateAttr
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

IMEND = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"

# 获取stop token的id
def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids

# 释放gpu显存
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

class agent(llms.LLM):
    _model_name: str = PrivateAttr()
    _tokenizer: AutoTokenizer = PrivateAttr()
    _generation_config: GenerationConfig = PrivateAttr()
    _stop_words_ids: List[int] = PrivateAttr()
    _sampling_params: SamplingParams = PrivateAttr()
    _model: LLM = PrivateAttr()
    def __init__(self, model_name):
        """一个模型作为agents的核心,生成对话"""
        super(agent, self).__init__()
        assert model_name in ["ChatGLM", "Qwen", "Baichuan"], "model_name should be in ['ChatGLM', 'Qwen', 'Baichuan']"
        self._model_name = model_name
        if model_name == "Qwen":
            model_dir = "pre_train_model/Qwen-7B-Chat"
            self._tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side='left', trust_remote_code=True)
            self._generation_config = GenerationConfig.from_pretrained(model_dir, pad_token_id=self._tokenizer.pad_token_id)
            self._stop_words_ids = []
            self._model = LLM(model=model_dir,
                                tokenizer=model_dir,
                                tensor_parallel_size=2,
                                trust_remote_code=True,
                                gpu_memory_utilization=0.8,
                                dtype="bfloat16")
            for stop_id in get_stop_words_ids(self._generation_config.chat_format, self._tokenizer):
                self._stop_words_ids.extend(stop_id)
            self._stop_words_ids.extend([self._generation_config.eos_token_id])
            sampling_kwargs = {
                "stop_token_ids": self._stop_words_ids,
                "early_stopping": False,
                "top_p": 1.0,
                "top_k": -1 if self._generation_config.top_k == 0 else self._generation_config.top_k,
                "temperature": 0.0,
                "max_tokens": 2000,
                "repetition_penalty": self._generation_config.repetition_penalty,
                "n":1,
                "best_of":2,
                "use_beam_search":True 
            }
            
            self._sampling_params = SamplingParams(**sampling_kwargs)
        if model_name == "ChatGLM":
            model_dir = "pre_train_model/glm4-9b"
            self._tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            self._model = LLM(model=model_dir,
                                tensor_parallel_size=2,
                                trust_remote_code=True,
                                max_model_len=1024,
                                gpu_memory_utilization=0.8,
                                enforce_eager=True,
                                dtype="bfloat16")
            stop_token_ids = [151329, 151336, 151338]
            self._sampling_params = SamplingParams(temperature=0.95, max_tokens=1024, stop_token_ids=stop_token_ids)

        if model_name == "Baichuan":
            self._model = LLM(model="pre_train_model/Baichuan2_7B_chat", tensor_parallel_size=2, trust_remote_code=True, gpu_memory_utilization=0.8, dtype="bfloat16")

    def _call(self,
        prompt: str,
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self._model_name == "Qwen":
            raw_text, _ = make_context(
                            self._tokenizer,
                            prompt,
                            system="You are a helpful assistant.",
                            max_window_size=self._generation_config.max_window_size,
                            chat_format=self._generation_config.chat_format,
                            )
            output = self._model.generate(raw_text,
                                        sampling_params = self._sampling_params
                                    )

            output_str = output[0].outputs[0].text
            if IMEND in output_str:
                output_str = output_str[:-len(IMEND)]
            if ENDOFTEXT in output_str:
                output_str = output_str[:-len(ENDOFTEXT)]
            torch_gc()
            return output_str
        if self._model_name == "ChatGLM":
            
            output = self._model.generate(prompts=prompt, sampling_params=self._sampling_params)
            torch_gc()
            return output[0].outputs[0].text
        
        if self._model_name == "Baichuan":
            output = self._model.generate(prompt, sampling_params = SamplingParams(temperature=0.95, max_tokens=1024))
            torch_gc()
            return output[0].outputs[0].text
    
    @property
    def _llm_type(self) -> str:
        """一个模型作为agents的核心,生成对话"""
        return f"custom:{self._model_name}"

if __name__ == "__main__":
    # qwen7 = "pre_train_model/Qwen-7B-Chat"
    # start = time.time()
    # llm = ChatLLM(qwen7)
    # print("model loaded")
    # test = ["吉利汽车座椅按摩","吉利汽车语音组手唤醒","自动驾驶功能介绍"]
    # generated_text = llm.infer(test)
    # print(generated_text)
    # end = time.time()
    # print("cost time: " + str((end-start)/60))
    start = time.time()
    llm = agent("Qwen")
    print("model loaded")
    print(llm.invoke("吉利汽车座椅按摩"))
    print(time.time()-start)
    # tokenizer = AutoTokenizer.from_pretrained("pre_train_model/glm4-9b", trust_remote_code=True)
    # print(tokenizer("你好", tokenize=False, add_generation_prompt=True))
