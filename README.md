# LLM_study
Learning LLM from Basic to Advanced

* model architecture 
    * Encoder Decoder
        * T5 model
    * Decoder
        * llama3  

本项目首先使用transformers对T5模型使用accelerate的训练，之后自己从0动手实现了llama3，并利用accelerate进行中文训练，从预训练到sft到DPO三个阶段全部完成，具体详情在对应文件夹下，通过本项目可以了解大模型的全流程训练。

* LLM inference
    * [Efficiently_Serving_LLMs笔记](inference_base/Efficiently_Serving_LLMs.md) 对应的笔记 记录了LLM相应的基础知识与技术，包括：高性能采样，动态批处理，Decoding Attention,Paged Attention,KV Cache,量化，矩阵乘量化
    * [Efficiently_Serving_LLMs.ipynb](inference_base/Efficiently_Serving_LLMs.ipynb) 使用了GPT2实现了动态批处理， KV cache,量化等技术