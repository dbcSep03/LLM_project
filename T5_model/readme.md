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
