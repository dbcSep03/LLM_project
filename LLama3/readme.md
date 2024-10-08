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

## Pretraining Training data
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

处理的代码路径为src/data/process_raw_data.py   
最终在fast总数据集大小为1.7GB(使用fastparquet的write)
在这个过程中遇到的问题：  
* 数据保存使用fastparquet 而读取使用pandas 默认读取出来为bytes
* 数据集1.7G pandas读取时间长   

解决如下
```python
import dask.dataframe as dd
import pandas as pd

start_time = time.time()
# 通过指定engine 解决了第一个问题
data = pd.read_parquet('LLama3/dataset/processed/pretrain_data.parquet', engine='fastparquet')
print(time.time() - start_time, len(data))
# 70s左右

start_time = time.time()
data = dd.read_parquet('LLama3/dataset/processed/pretrain_data.parquet')
print(time.time() - start_time, len(data))
# 0.2s
```
觉得dask.dataframe效果很好 而且相较于pandas,内存消耗更小 
64G的内存 当读取的时候，内存占用可以到40G,很奇怪

> 我又到了晚上重新尝试了一下 pd.read_parquet('LLama3/dataset/processed/pretrain_data.parquet')可以正常读了，没有b'[]'的形式 小小的脑袋大大的疑惑
## Model Pretrain
预训练使用了两种方式，一个是accelerate方式实现、一个是torchrun的方式
### accelerate方式
使用accelerate包装了一下 在运行之前，需要使用```accelerate config``` 设置一下。如果想要多卡运行，需要使用```accelerate launch --multi-gpu {.py} ```    
loss图如下：
![accelerate预训练](img/accelerate-pretrain.png)     

由于是双卡，我将两张卡的loss都保存下来了，感觉不如使用tranformers的loss损失平滑，可能有没注意到的细节。

演示效果：
![accelerate演示](img/accelerate-output.gif)

很明显有以下问题 需要解决：
* 输出重复
* 前后语境不搭
* \<eos>未结束， tokenizer的时候有空格  
  * 该问题得到了解决，是因为在数据集预处理的时候采用了\<eos>但是tokenizer为<EOS>数据集需要重新处理，重新训练  

重新训练的结果如下

![演示效果](img/accelerate-repretrain.gif)

* 当使用正确的\<EOS>分词技术可以解决语境问题 取出来特殊的tokenizer,还剩下输出重复问题，可以使用生成时的一些生成track解决

### torchrun 双卡分布训练    
机器为i7-12700KF+64G+双卡4090
训练的时候遇到了两个问题:    
* ```train_dataloader = DataLoader(dataset, batch_size=trainConfig.batch_size // world_size, shuffle=False, collate_fn=dataset.collate_fn, num_workers=4, sampler=sampler)```会出现炸内存的情况，数据集2G 内存64G，会直接超出。(不是显存)    
  * [num_workers>0的问题](https://github.com/pytorch/pytorch/issues/13246) 通过查阅，发现是dataloader的加载的问题   
* 训练速度 accelerate在bs=12的情况下，单卡显存为14G，但是一个epoch7h，而torchrun在bs=12时，显存为10G，但是一个epoch在72h多，很奇怪      
  * 我仔细想了一下，可能是我在accelerate config中设置了bf16,改完之后，并没有变得那么快，还是六十多个小时，但是显存降到了8.4G    
  * 感觉更加奇怪的是，我将双卡torchrun(pretrain_pytorch.py)代码改为单卡训练(pretrain_model.py),bs=12时，只需要12h(默认float32),就算没有连接桥，也不应该那么慢，很amazing    
  * 最终选择使用单卡训练 而不是双卡


## SFT
### SFT Dataset 
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
### SFT Train
再次尝试了pandas和datasets，感觉pandas要比datasets节省内存。 使用命令 ```accelerate launch --multi-gpu {.py} ``` 代码在sft_accelerate.py

### LoRA_SFT
从零实现一个LoRA,本质上使用矩阵模拟权重的变化$\Delta w$,使用A@B来进行模型  
在从普通的线性层修改成LoRA层，使用了递归的方法，model.named_children()和setattr(object, name, value)两个部分  
模型架构从
```
LLamamodel(
  (embedding): Embedding(30001, 1024, padding_idx=6)
  (embedding_rmsnorm): RMSNorm()
  (decoder): ModuleList(
    (0-9): 10 x LLamaDecoder(
      (input_norm): RMSNorm()
      (self_attn): LLamaAttention(
        (rotary_embedding): LlamaRotaryEmbedding()
        (q): Linear(in_features=1024, out_features=1024, bias=False)
        (k): Linear(in_features=1024, out_features=512, bias=False)
        (v): Linear(in_features=1024, out_features=512, bias=False)
        (o): Linear(in_features=1024, out_features=1024, bias=False)
      )
      (mlp): LLamaMLP(
        (gate_proj): Linear(in_features=1024, out_features=2048, bias=False)
        (up_proj): Linear(in_features=1024, out_features=2048, bias=False)
        (down_proj): Linear(in_features=2048, out_features=1024, bias=False)
        (act_fn): SiLU()
      )
      (post_attention_layernorm): RMSNorm()
    )
  )
  (norm): RMSNorm()
  (lm_head): Linear(in_features=1024, out_features=30001, bias=False)
)
```
变成了
```
LoRA_model(
  (model): LLamamodel(
    (embedding): Embedding(30001, 1024, padding_idx=6)
    (embedding_rmsnorm): RMSNorm()
    (decoder): ModuleList(
      (0-9): 10 x LLamaDecoder(
        (input_norm): RMSNorm()
        (self_attn): LLamaAttention(
          (rotary_embedding): LlamaRotaryEmbedding()
          (q): LinearLoRA(
            (linear): Linear(in_features=1024, out_features=1024, bias=False)
            (lora): LoRALayer()
          )
          (k): Linear(in_features=1024, out_features=512, bias=False)
          (v): LinearLoRA(
            (linear): Linear(in_features=1024, out_features=512, bias=False)
            (lora): LoRALayer()
          )
          (o): Linear(in_features=1024, out_features=1024, bias=False)
        )
        (mlp): LLamaMLP(
          (gate_proj): LinearLoRA(
            (linear): Linear(in_features=1024, out_features=2048, bias=False)
            (lora): LoRALayer()
          )
          (up_proj): LinearLoRA(
            (linear): Linear(in_features=1024, out_features=2048, bias=False)
            (lora): LoRALayer()
          )
          (down_proj): LinearLoRA(
            (linear): Linear(in_features=2048, out_features=1024, bias=False)
            (lora): LoRALayer()
          )
          (act_fn): SiLU()
        )
        (post_attention_layernorm): RMSNorm()
      )
    )
    (norm): RMSNorm()
    (lm_head): Linear(in_features=1024, out_features=30001, bias=False)
  )
)
```
微调了['q', 'v', 'up_proj', 'gate_proj', 'down_proj']，超参数为 r=8,alpha=16，一样使用accelerate 进行双卡训练

### SFT results
SFT阶段，全参微调和lora微调的结果如下
![SFT-finetuning/lora](img/sft_lora.png) 

可以看到finetuning的loss损失比lora要低，全参微调的优势还是存在的
### 演示效果
下图是全参微调的结果
![sft](img/sft.gif)

下图是lora微调的效果
![lora](img/lora.gif)

经过微调，解决了一定的上下文语境的问题。我的理解是在预训练阶段没有padding的token，会将上下文的语境全部连续起来，而在sft阶段通过padding的存在，解决了一个序列只有一个语境，同时输出重复问题得到了改善，虽然这两次演示没有，但经过多次检测，还是会有该问题。可以通过改良generate的方法来对生成的方式提优。

## DPO
对强化学习有点不熟，先恶补了一下强化学习的相关知识，首先是强化学习的知识，[笔记](强化学习笔记.md)，然后是学习了RLHF的PPO方法到DPO方法的转换，[RLHF笔记](RLHF.md)，最终亲手实现了DPO方法。  
核心公式为 $$\max_{\pi_\theta} \left\{ \mathbb{E}_{(x, y{\text{win}}, y_{\text{lose}}) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_{\text{win}}|x)}{\pi_{\text{ref}}(y_{\text{win}}|x)} - \beta \log \frac{\pi_\theta(y_{\text{lose}}|x)}{\pi_{\text{ref}}(y_{\text{lose}}|x)} \right) \right] \right\}$$
在实现的过程中觉得有意思点是：如何只对answer部分选取真实标签进行loss mask,最终利用tokenizer的token_type_id部分，prompt和padding为0，而answer部分为1，通过answer*input_id实现了prompt和padding任然为0，但是answer部分为真实id，使用torch.gather方法选取对应的token的logits。   
loss部分如下：  
![DPO_margins](img/DPO_margins.png)
![DPO_reward_acciracoes](img/DPO_reward_accuracies.png)  

在单卡2080ti上训练，bs=1,梯度累积为8，从图中可以看出：奖励的margins越来越大且将奖励正确的接近8，可以说起到了训练的效果。  
训练脚本在LLama3/dpo.py。


> 相关资料   
> 分词化：[BPE](https://github.com/karpathy/minbpe)   
> LLama3 from scratch: [LLama3 from scratch](https://github.com/naklecha/llama3-from-scratch) (阿尼亚很可爱)   
> RoPE: [知乎讲解 十分钟读懂旋转编码（RoPE）](https://www.zhihu.com/tardis/zm/art/647109286?source_id=1003) 
> LoRA from scratch: [LoRA from scratch](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch)