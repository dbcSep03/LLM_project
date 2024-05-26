# T5

此项目仿照 [ChatLM-mini-Chinese](https://github.com/charent/ChatLM-mini-Chinese?tab=readme-ov-file)  
从预训练到微调，使用DDO，包含数据预处理   
## T5 architecture
* T5 是encoder-decoder的架构，从文本到文本的生成模型  
  * 即encoder出来的特征作为key和value， decoder出来的特征作为query进行attention操作
* position embedding
  * 不是直接加在特征上 而是在每一层加入
* 计算loss忽略的-100的token  
  * 即在计算batch进行padding的时候，padding_label=-100 让其不进行 
## Dataset
数据仿照[ChatLM-mini-Chinese](https://github.com/charent/ChatLM-mini-Chinese?tab=readme-ov-file)  
改动如下：  
* 没有使用医药领域数据集
* belle数据集没有找到Belle_open_source_1M
* Wikipedia使用了 [zhwiki](https://dumps.wikimedia.org/zhwiki/)   
  * 使用了20240520下的Recombine articles, templates, media/file descriptions, and primary meta-pages. 共2.4GB
  * bz2转换为wiki.txt参考了 [WikiExtractor](https://github.com/apertium/WikiExtractor) 
  
## Pre_tokenizer
只使用Wikipedia的简体语料，作为分词的数据集
本项目只采用了BPE分词，处理如下   
*  进行NFKC()正则化
*  pre_tokenizer为符号分开 数字分开(单独) 并且按照空格分开且替换为_   

数据集总大小为1.6GB 在62.6G上 i7-12700KF，一次训练整个语料会出现swap,且非常卡。   
选择进行buffer迭代缓冲进行训练，且训练的tokenizer保存为两个class 一种是tokenizer，一个是PreTrainedTokenizerFast

## Model Train(Transformers)
模型训练，我理解共分为以下几个步骤   
* Tokenizer的实例化
* ModelConfig和模型的实例化
* 数据集的加载 包括加载、分词化(同时进行padding和截断)
* 进行训练参数的实例化和collator
* 模型训练

在项目[ChatLM-mini-Chinese](https://github.com/charent/ChatLM-mini-Chinese?tab=readme-ov-file)中，虽然在collator中选择了max_length,但是padding=True 所以无效了，我最终选择在数据集加载中，处理数据集

## Model Train(PyTorch)
使用了自己实现的T5model，在训练和评估的时候 都使用了交叉熵做为loss进行计算   
风格为使用pytorch实现的模型训练方式   
代码为pre_train_model.py 模型路径在src/models下面 包含了自己实现的Dataset和model
