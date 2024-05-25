import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from transformers import PreTrainedTokenizerFast
from config import ModelConfig
import numpy as np
class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(T5LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epos = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x*torch.rsqrt(variance + self.epos)
        return self.weight * x

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_kv, has_relative_attention_bias=False, relative_attention_num_buckets=32):
        super(Attention, self).__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.has_relative_attention_bias = has_relative_attention_bias
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)
        assert d_kv == d_model//num_heads
        self.head_size = num_heads
        self.d_kv = d_kv
        self.d_model = d_model

    def shape(self, x):
        return x.contiguous().view(x.shape[0], -1, self.head_size, self.d_kv).transpose(1,2)

    def unshape(self, x):
        return x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

    @staticmethod
    def _relative_position_bucket(query_length, device, bidirectional=True, num_buckets=32, max_distance=128):

        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(query_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)

        return relative_buckets

    def forward(self, hidden_states, attention_masks=None, encoder_hidden_state=None, encoder_attention_mask=None):
        if encoder_attention_mask is not None and attention_masks is not None:
            raise ValueError("Cannot specify both encoder_attention_mask and attention_masks")
        q = self.shape(self.q(hidden_states))
        if encoder_hidden_state is not None:
            k = self.shape(self.k(encoder_hidden_state))
            v = self.shape(self.v(encoder_hidden_state))
        else:
            k = self.shape(self.k(hidden_states))
            v = self.shape(self.v(hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2))


        if self.has_relative_attention_bias:
            relative_position_bucket = self._relative_position_bucket(
                query_length=hidden_states.shape[1], device=self.relative_attention_bias.weight.device)
            position_bias = self.relative_attention_bias(relative_position_bucket).permute([2, 0, 1]).unsqueeze(0)
            attention_masks = position_bias + attention_masks

        if attention_masks is not None:
            scores += attention_masks
        attention_weights = F.softmax(scores/torch.full([], self.d_kv**2, device=self.q.weight.device), dim=-1)
        attn_output = self.unshape(torch.matmul(attention_weights, v))
        attn_output = self.o(attn_output)
        return attn_output, attention_masks


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_kv, has_relative_attention_bias, dropout):
        super(SelfAttention, self).__init__()
        self.layernorm = T5LayerNorm(d_model)
        self.attention = Attention(d_model, num_heads, d_kv, has_relative_attention_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_masks=None):
        normed_hidden_states = self.layernorm(hidden_states)
        attention_output, attention_masks = self.attention(normed_hidden_states, attention_masks)
        hidden_states = hidden_states + self.dropout(attention_output)
        return hidden_states, attention_masks
    
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_kv, dropout):
        super(CrossAttention, self).__init__()
        self.LayerNorm = T5LayerNorm(d_model)
        self.attention = Attention(d_model, num_heads, d_kv)
        self.dropout = nn.Dropout(dropout)
    def forward(self, hidden_states, encoder_hidden_state, encoder_attention_mask):
        decoder_hidden_states = self.LayerNorm(hidden_states)
        decoder_hidden_states, encoder_attention_mask = self.attention(decoder_hidden_states,
                                                                       attention_masks=encoder_attention_mask,
                                                                       encoder_hidden_state=encoder_hidden_state)
        hidden_states = hidden_states + self.dropout(decoder_hidden_states)
        return hidden_states, encoder_attention_mask

class FFn(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FFn, self).__init__()
        self.layernorm = T5LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x):
        h_ = self.layernorm(x)
        h_ = self.fc1(h_)
        h_ = self.dropout1(F.relu(h_))
        h_ = self.fc2(h_)
        return x + self.dropout2(h_)


class T5Block(nn.Module):
    def __init__(self, is_decoder, d_model, num_heads, d_kv, d_ff, has_relative_attention_bias, dropout):
        super(T5Block, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SelfAttention(d_model, num_heads, d_kv, has_relative_attention_bias, dropout))
        self.is_decoder = is_decoder

        if is_decoder:
            self.layers.append(CrossAttention(d_model, num_heads, d_kv, dropout))
        self.layers.append(FFn(d_model, d_ff, dropout))

    def forward(self, hidden_states, attention_masks=None, encoder_hidden_state=None, encoder_attention_mask=None):
        hidden_states, attention_masks = self.layers[0](hidden_states, attention_masks)
        if self.is_decoder and encoder_hidden_state is not None:
            hidden_states, encoder_attention_mask = self.layers[1](hidden_states, encoder_hidden_state, encoder_attention_mask)

        hidden_states = self.layers[-1](hidden_states)
        return hidden_states, attention_masks, encoder_hidden_state, encoder_attention_mask


class T5model(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_kv, d_ff, num_encoder_layers, num_decoder_layers, dropout, bos_id, padding_id):
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.ModuleList([
            T5Block(is_decoder=False, d_model=d_model, num_heads=num_heads, d_kv=d_kv, d_ff=d_ff,
                    has_relative_attention_bias=bool(i==0), dropout=dropout)
            for i in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([
            T5Block(is_decoder=True, d_model=d_model, num_heads=num_heads, d_kv=d_kv, d_ff=d_ff,
                    has_relative_attention_bias=bool(i==0), dropout=dropout
                    ) for i in range(num_decoder_layers)])

        self.dropout = nn.Dropout(dropout)
        self.bos_id = bos_id
        self.padding_id = padding_id
        self.d_model = d_model
        self.lm_head = nn.Linear(d_model, vocab_size)
    def right_shift(self, labels):
        decoder_input = labels.new_zeros(labels.shape)
        decoder_input[:, 1:] = labels[:, :-1].clone()
        decoder_input[:, 0] = self.bos_id
        decoder_input.masked_fill_(decoder_input == -100, self.padding_id)
        return decoder_input

    def forward(self, input_ids, attention_masks, labels):
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.dropout(hidden_states)

        if len(attention_masks.shape) == 2:
            attention_masks = (1.0 - attention_masks) * torch.finfo(hidden_states.dtype).min
            attention_masks = attention_masks[:, None, None, :]

        _attention_masks = attention_masks
        for layer in self.encoder:
            hidden_states, attention_masks, _, _ = layer(hidden_states, attention_masks)

        decoder_input = self.right_shift(labels)
        decoder_hidden_states = self.embeddings(decoder_input)
        decoder_hidden_states = self.dropout(decoder_hidden_states)

        attention_masks = torch.tril(torch.ones(decoder_hidden_states.shape[1], decoder_hidden_states.shape[1], dtype=torch.long, device=decoder_hidden_states.device))
        attention_masks = (1.0 - attention_masks) * torch.finfo(decoder_hidden_states.dtype).min
        attention_masks = torch.tensor(attention_masks[None, None, :,:])
        for layer in self.decoder:
            decoder_hidden_states, attention_masks, _,_ = layer(
                hidden_states=decoder_hidden_states, attention_masks=attention_masks,
                encoder_hidden_state=hidden_states, encoder_attention_mask=_attention_masks)
        sequence_output = decoder_hidden_states * (self.d_model ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        return lm_logits


    def train(self, input_ids, attention_masks, labels, criterion):
        logits = self.forward(input_ids, attention_masks, labels)
        labels = labels.to(logits.device)
        logits = logits.view(-1, self.vocab_size)
        loss = criterion(logits, labels.view(-1))
        return loss

if __name__ == "__main__":
    ques = ["你读书的理由是什么[EOS]", "请简单介绍一下大连理工大学?谢谢您[EOS]"]
    ans = ["我爱我的祖国[EOS]", "大连理工最近建了理塘，它由校友出资,是世界的最高点[EOS]"]

    tokenizer = PreTrainedTokenizerFast.from_pretrained('./tokenizer_fast')
    input = tokenizer(ques, padding=True)
    ans = tokenizer(ans, padding=True)

    input_ids = np.array(input['input_ids'], np.int64)
    attention_masks = np.array(input['attention_mask'], np.int64)
    labels = np.array(ans['input_ids'], np.int64)
    input_ids = torch.LongTensor(input_ids)
    attention_masks = torch.LongTensor(attention_masks)
    labels = torch.LongTensor(labels)
    labels[labels == tokenizer.pad_token_id] = -100

    model = T5model(len(tokenizer.get_vocab()),
                    d_model=ModelConfig.d_model, num_heads=ModelConfig.num_heads, d_kv=ModelConfig.d_kv,
                    d_ff=ModelConfig.d_ff,num_encoder_layers=ModelConfig.num_layers,
                    num_decoder_layers=ModelConfig.num_decoder_layers,dropout=0.1,
                    bos_id=tokenizer.bos_token_id, padding_id=tokenizer.pad_token_id)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    loss = model.train(input_ids, attention_masks, labels, criterion)




