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
    * [vllm_inference](inference_base/vllm_inference)使用了vllm，对QWEN1.5-7B进行了推理，将单卡推理/int4AWQ/FP8KV-cache/双卡进行了测试，并测试了在单条，4，8，16，32下吞吐量。
* LLM Agent  
    * [langchain/RAG_base.ipynb](langchain/RAG_base.ipynb) 对应于LangChain的相关知识：query translation\query construction\Routing\Indexing\Retrieval\Genration的步骤
    * [langchain/RAG_Agent](langchain/RAG_Agent) 包含了一个实际项目，根据现有的车主手册构建知识库，进行知识问答
        * 构建知识库：使用pdfplumber对pdf进行解析，包含了两种解析：分块解析、滑窗解析
        * 知识检索：使用了BGE检索和GTE检索，并在检索的基础上使用了Reranker技术
        * 答案生成：Agent基座使用Qwen,并可以选配ChatGLM,Baichuan2,使用Agent对检索文档进行总结后，使用Answer recursively不断优化升级后的答案
* LLM NER
    * [Medicine_LLM](Medicine_LLM) 想法来源于天池比赛--[中文医疗大模型评测](https://tianchi.aliyun.com/competition/entrance/532085/information)   
        * 使用真实的医疗数据集，将命名实体识别等分类任务转变成文本生成任务
        * 使用QloRA微调QWEN1.5-7B-Chat,并使用prompt templates构建结构化输入输出
        * 仅仅对5200条的1epoch的微调，可在测试650条上达到10个种类的F1接近1。
