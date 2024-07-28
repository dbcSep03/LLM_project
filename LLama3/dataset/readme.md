# 数据集
## Pretrain data
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

## SFT data
  * [BELLE Group](https://huggingface.co/BelleGroup)   
    * 包含了[generated_chat_0.4M](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M) 和[train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
    * 总中文字符为274882143
  * [shareAI 数据集](https://modelscope.cn/datasets/baicai003/Llama3-Chinese-dataset/summary)
    * 感觉数据质量本身非常优秀，同时作为SFT数据集
    * 和pretrained_data一样 总中文字符为455168273
  * [m-a-p/COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA)
    * shareAI数据集中包括，但感觉不完全，使用了全部
    * 总中文字符为48014083
  * [lyuricky/alpaca_data_zh_51k](https://huggingface.co/datasets/lyuricky/alpaca_data_zh_51k)
    * alpaca数据集的中文
    * 总中文字符为5477887

## DPO data
* [Skepsun/huozi_rlhf_data_json](https://huggingface.co/datasets/Skepsun/huozi_rlhf_data_json?row=11) 
  * [活字通用大模型](https://github.com/HIT-SCIR/huozi)的对齐数据集
* [beyond/rlhf-reward-single-round-trans_chinese](https://huggingface.co/datasets/beyond/rlhf-reward-single-round-trans_chinese)  
  * prompt chosen 和rejected三个部分
  * 使用了对应的train数据集，同时文件重命名为rlhf-reward-single-round-trans_chinese。
> 感觉RLHF的数据集质量不太行，在重要名词部分掺杂着部分英文


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