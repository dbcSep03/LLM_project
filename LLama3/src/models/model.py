import torch
import torch.nn as nn
from typing import Optional
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # nn.Parameter()定义需要在训练过程中进行优化的参数 让embedding每个维度在缩放后也可以进行单独的进行缩放
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.type(torch.float32)
        mean = hidden_states.pow(2).mean(-1, keepdim=True) # 链式访问写法更符合面向对象编程 
        # 考虑过 torch.mean(torch.pow(hidden_states, 2), -1, keepdim=True) 和 torch.mean(hidden_states**2, -1, keepdim=True) 
        hidden_states = hidden_states * torch.rsqrt(mean+self.eps)
        return self.weight * hidden_states.type(dtype)

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, head_size, config):
        super().__init__()
        self.size = head_size
        self.max_position_embeddings = config.seq_length
        self.base = config.rotary_base
        # 1/10000^(2i/d) 2i/d是因为sin和cos两个维度 将2个看作一组  还有个问题 self.inv_freq 不能直接等于 这样需要单独的加载和保存
        # self.inv_freq = 1 / (self.base ** (torch.arange(0, self.size, 2, dtype=torch.int64).float()/self.dim))
        inv_freq = 1 / (self.base ** (torch.arange(0, self.size, 2, dtype=torch.float32)/self.size))
        self.register_buffer("inv_freq", inv_freq)
    
    @torch.no_grad()
    def forward(self,x ,position_ids):
        # x: [bsz, seq_length, head_size]
        # position_ids: [bsz, seq_length]
        # .float()：涉及到sin cos 保证计算精度
        # inv_freq: [head_size//2] -> [1, head_size//2, 1] -> [bsz, head_size//2, 1]
        inv_freq_expand = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids: [bsz, 1, seq_length]
        position_ids_expand = position_ids[:,None,:].float()
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            """防止精度混合"""
            # 由于@进行矩阵计算  [bsz, head_size//2, 1] * [bsz, 1, seq_length] -> [bsz, head_size//2, seq_length] -> [bsz, seq_length, head_size//2]
            freqs = (inv_freq_expand.float() @ position_ids_expand.float()).transpose(1, 2)
            # [bsz, seq_length, head_size//2] -> [bsz, seq_length, head_size]
            emb = torch.cat((freqs,freqs),dim=-1)
            sin = emb.sin()
            cos = emb.cos()
        return sin.to(x.dtype), cos.to(x.dtype)

def applay_rotary_pos_emb(query, key, sin, cos, unsqueeze_dim = 1):
    def rotate_half(x):
        # x: [bsz, seq_length, num_heads, head_size]
        # x1: [bsz, seq_length, num_heads, head_size//2] x2:[bsz, seq_length, num_heads, head_size//2]
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # 非常巧妙 在二维的时候 公式为 x_0 = cos\theta * x_0 - sin\theta * x_1  x_1 = cos\theta * x_0 + sin\theta * x_1
    # 看上去是x_0 x_1 拓展到 head_size时候 可以理解为 前 head_size//2 为实数部 后 head_size//2为复数部
    # 和torch.cat((freqs, freqs), dim=-1)对应
    q_embed = (query * cos) + (rotate_half(query) * sin)
    k_embed = (key * cos) + (rotate_half(key) * sin)
    return q_embed, k_embed

def repeat_kv(kv, num_groups):
    if num_groups == 1:
        return kv
    bsz, num_qv_heads, seq_length, head_size = kv.size()
    # [bsz, num_qv_heads, seq_length, head_size] -> [bsz, num_qv_heads, num_groups, seq_length, head_size]
    kv = kv[:, :, None, :, :].expand(bsz, num_qv_heads, num_groups, seq_length, head_size)
    # [bsz, num_qv_heads, num_groups, seq_length, head_size] -> [bsz, num_qv_heads * num_groups, seq_length, head_size]
    # 维度对齐
    return kv.contiguous().view(bsz, num_qv_heads * num_groups, seq_length, head_size)

class LLamaAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        
        self.head_size = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.rotary_embedding = LlamaRotaryEmbedding(self.head_size, config)

        self.q = nn.Linear(self.hidden_size, self.head_size * self.num_heads, bias=config.bias)
        self.k = nn.Linear(self.hidden_size, self.head_size * self.num_key_value_heads, bias=config.bias)
        self.v = nn.Linear(self.hidden_size, self.head_size * self.num_key_value_heads, bias=config.bias)
        self.o = nn.Linear(self.hidden_size, self.hidden_size, bias=config.bias)
    def forward(self, 
                input_ids: torch.LongTensor, 
                attention_mask: torch.LongTensor, 
                position_ids: torch.LongTensor):
        bsz, seq_length, _ = input_ids.size()

        query_states = self.q(input_ids).view(bsz, seq_length, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        key_states = self.k(input_ids).view(bsz, seq_length, self.num_key_value_heads, self.head_size).permute(0, 2, 1, 3)
        value_states = self.v(input_ids).view(bsz, seq_length, self.num_key_value_heads, self.head_size).permute(0, 2, 1, 3)
        
        sin , cos = self.rotary_embedding(input_ids, position_ids)
        query_states, key_states = applay_rotary_pos_emb(query_states, key_states, sin, cos)
        # https://github.com/meta-llama/llama/issues/384 
        # 关于group attention, 可以看看上面的链接 但是有问题在于图中 keys和query 就应该两两相连
        # 我的理解是 如果不repeat的话 维度有问题 无法相乘 
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))/ (self.head_size**0.5)

        if attn_weights is not None:
            causal_mask = attention_mask[:, : , : , : key_states.size(-2)]
            attn_weights = attn_weights + causal_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.config.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1,2).contiguous().view(bsz, seq_length, self.hidden_size)
        attn_output = self.o(attn_output)

        return attn_output
        


class LLamaMLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
        self.act_fn = nn.SiLU()
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x) * self.up_proj(x)))


class LLamaDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_norm = RMSNorm(config.hidden_size, eps=config.eps)
        self.self_attn = LLamaAttention(config)
        self.mlp = LLamaMLP(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.eps)
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor, 
                position_ids: torch.Tensor):
        # attention
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = hidden_states + residual
        
        # mlp
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        output = self.mlp(hidden_states)
        return output + residual

class LLamamodel(nn.Module):
    def __init__(self,config):
        super(LLamamodel, self).__init__()
        self.vocab_size = config.vocab_size
        self.padding_idx = config.padding_idx
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, config.padding_idx)

        self.embedding_rmsnorm = RMSNorm(config.hidden_size)

        self.decoder = nn.ModuleList([
            LLamaDecoder(config) for _ in range(config.layer)
        ])
        
        self.norm = RMSNorm(config.hidden_size)
        
        # 直接将hidden_size映射到vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, 
                input_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor]=None,
                position_ids: Optional[torch.LongTensor]=None,
                whether_get_loss: bool = False):
        """
        为了学习deepspeed accelerator的使用
        多了个whether_get_loss参数
        是否在内部计算loss
        由于是Forcausal的结构 所以 我就直接取消了labels的传入
        """
        hidden_states = self.embedding(input_ids)
        hidden_states = self.embedding_rmsnorm(hidden_states)
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
        
        causal_mask = self._update_causal_mask(attention_mask, hidden_states)
        
        for decoder in self.decoder:
            hidden_states = decoder(hidden_states, causal_mask, position_ids)
        hidden_states = self.norm(hidden_states)

        logits = self.lm_head(hidden_states)
        
        loss = None
        # 接下来计算loss 
        if whether_get_loss:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()

            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)
        return logits, loss

            
        
    def  _update_causal_mask(self, attention_mask, hidden_states):
        dtype, device = hidden_states.dtype, hidden_states.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = hidden_states.shape[1]
        if attention_mask is not None:
            return attention_mask
        else:
            causal_mask = torch.full((sequence_length, sequence_length),fill_value=min_dtype, dtype=dtype, device=device)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            # expand中-1表示维度不进行变化
            causal_mask = causal_mask[None, None, :, :].expand(hidden_states.shape[0], 1, -1, -1)
        return causal_mask
    
    @torch.inference_mode()
    def generate(self, input_ids, attention_masks=None, max_length=512):
        generated = input_ids
        for _ in range(max_length):
            position_ids = torch.arange(0,generated.size(1), device=generated.device).unsqueeze(0)
            logits, _ = self.forward(generated, attention_masks, position_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == self.config.eos_token_id:
                break
        return generated
        
