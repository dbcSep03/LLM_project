import torch.nn as nn
import torch
from typing import Optional
class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, r, alpha):
        super().__init__()
        std_dev = 1/torch.sqrt(torch.tensor(r).float())  # 缩放根号r
        self.A = nn.Parameter(torch.randn(input_dim, r) * std_dev)
        self.B = nn.Parameter(torch.zeros(r, output_dim))
        self.alpha = alpha
    
    def forward(self,x):
        x = self.alpha * (x @ self.A @self.B) # 矩阵的分解
        return x

class LinearLoRA(nn.Module):
    def __init__(self, linear, r, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, r, alpha)
    
    def forward(self, x):
        x = self.linear(x) + self.lora(x)
        return x

class LoRA_model(nn.Module):
    def __init__(self, model, r, alpha, lora_name, eos_token_id=None):
        super().__init__()
        self.model = model
        self.eos_token_id = eos_token_id
        for param in self.model.parameters():
            param.requires_grad = False
        self.replace_specific_linear_with_lora(self.model, lora_name, r, alpha)
    
    def replace_specific_linear_with_lora(self, model, lora_name, r, alpha):
        for name, module in model.named_children():
            if isinstance(module, nn.Linear) and name in lora_name:
                setattr(model, name, LinearLoRA(module, r, alpha))
            else:
                self.replace_specific_linear_with_lora(module, lora_name, r, alpha)
        
    
    def forward(self, x,
                attention_mask: Optional[torch.Tensor]=None,
                position_ids: Optional[torch.LongTensor]=None,
                whether_get_loss: bool = False):
        return self.model(x,attention_mask,position_ids,whether_get_loss)

    @torch.inference_mode()
    def generate(self, input_ids, attention_masks=None, max_length=512):
        generated = input_ids
        for _ in range(max_length):
            position_ids = torch.arange(0,generated.size(1), device=generated.device).unsqueeze(0)
            logits, _ = self.forward(generated, attention_masks, position_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == self.eos_token_id:
                break
        return generated