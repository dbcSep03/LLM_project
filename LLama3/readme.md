# LLama3
llama2和llama3好像没有什么架构上的区别,主要看了transformers的LlamaForCausalLM实现，进行了架构上的复现   
## model architecture
* 激活函数使用了 SiLU 
* 位置编码是RoPE 挺有意思的 理论和代码还是挺有区别的 特别是在维度上的处理
* mlp没有dropout

> attention_mask和loss_mask的启发：[baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)  
> 和小伙伴讨论了一下，有以下的启发：
> 无论是pre_trained 和 sft的阶段 无论是否padding   
> 文本处理：pretrained直接文本拼接 不同文本用<eos>连接 而在padding阶段为<question>+<bos>+<answer>+<eos>+padding 到最大序列   
> attention_mask:都使用下三角和对角线为0 上三角为-inf的矩阵，因为核心理解为attention_mask 序列逐一预测 $\sum log \pi(t_i|t_{i-1}...t_0)$
> loss_mask:pretrained直接计算loss 而sft对padding和prompt进行系数掩码 系数为0

## Training data
* Wikipedia使用了 [zhwiki](https://dumps.wikimedia.org/zhwiki/)   
  * 使用了20240520下的Recombine articles, templates, media/file descriptions, and primary meta-pages. 共2.4GB
  * bz2转换为wiki.txt参考了 [WikiExtractor](https://github.com/apertium/WikiExtractor) 
  * 只用Wikipedia来训练tokenizer 
  * 可以参考T5_model/src/data的数据预处理
* [firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
  * 收集了23个常见的中文数据集，对于每个任务，由人工书写若干种指令模板，保证数据的高质量与丰富度，数据量为115万 
* [Llama3中文数据集](https://modelscope.cn/datasets/baicai003/Llama3-Chinese-dataset/files)
  * shareAI整理的数据集
  * 其中1_0_firefly_chinese_common_task_1649k.jsonl 重复了 我删除了该项
* [ruozhiba](https://huggingface.co/datasets/LooksJuicy/ruozhiba)   
  * 经典弱智吧    

处理的


   
> 相关资料   
> 分词化：[BPE](https://github.com/karpathy/minbpe)   
> LLama3 from scratch: [LLama3 from scratch](https://github.com/naklecha/llama3-from-scratch) (阿尼亚很可爱)   
> RoPE: [知乎讲解 十分钟读懂旋转编码（RoPE）](https://www.zhihu.com/tardis/zm/art/647109286?source_id=1003) 