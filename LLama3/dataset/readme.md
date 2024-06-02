# 数据集
## Pretrain data process
* [shareAI 数据集](https://modelscope.cn/datasets/baicai003/Llama3-Chinese-dataset/summary)
  * 本身是由sft_all_data的，我用来作为预训练数据集
  * 他的前两个数据集 0_english_* 是英文的 我希望做一个中文LLama3 所以给去除了
  * 处理之后，总中文字符为455168273
* [ruozhiba](https://huggingface.co/datasets/LooksJuicy/ruozhiba)
  * 使用了该数据集 而不是其他两个gpt-4o和gpt-4的数据集 感觉生成的没有对应的魅力
  * 处理之后中文字符为173830
* [wiki_simple.txt](https://dumps.wikimedia.org/zhwiki/20240520/)
  * 使用了20240520的版本
  * 中文字符为297557817
* belle 是开源的指令训练数据 
  * 为[BELLE Group](https://huggingface.co/BelleGroup) 
  * 包含了Belle_open_source_1M.json\school_math_0.25M.json\train_2M_CN.json
  * 总中文字符数为832765052

## SFT data process
  * [BELLE Group](https://huggingface.co/BelleGroup)   
    * 包含了[generated_chat_0.4M](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M) 和[train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
  * [shareAI 数据集](https://modelscope.cn/datasets/baicai003/Llama3-Chinese-dataset/summary)
    * 感觉数据质量本身非常优秀，同时作为SFT数据集
  * [m-a-p/COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA)
    * shareAI数据集中包括，但感觉不完全，使用了全部
  * [lyuricky/alpaca_data_zh_51k](https://huggingface.co/datasets/lyuricky/alpaca_data_zh_51k)
    * alpaca数据集的中文
  
## 期望的数据结构
```
-dataset
  -processed
    # 处理好的数据集
  
  -raw
    # 未处理的原数据集
    - alpaca_data_zh_51K
    -belle_pretrain
    -belle_sft
    -COIG-CQIA
    -ruozhiba
    -shareAI
    -zhwiki
  
  -sft
    # 微调数据集
  
```