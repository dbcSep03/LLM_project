import torch.nn as nn
import torch
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
    def __init__(self, model, r, alpha, lora_name):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.replace_specific_linear_with_lora(self.model, lora_name, r, alpha)
    
    def replace_specific_linear_with_lora(self, model, lora_name, r, alpha):
        for name, module in model.named_children():
            if isinstance(module, nn.Linear) and name in lora_name:
                setattr(model, name, LinearLoRA(module, r, alpha))
            else:
                self.replace_specific_linear_with_lora(module, lora_name, r, alpha)
        
    
    def forward(self, x):
        return self.model(x)