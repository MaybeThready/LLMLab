# -*- coding: UTF-8 -*-

import torch
import math


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

counts = 0
def replace_linear_with_lora(network, rank, alpha, device, module_names=None, tag=-1):
    global counts
    if tag == -1:
        counts = 0
    for name, module in network.named_children():
        if module_names is None:
            if isinstance(module, torch.nn.Linear):
                try:
                    setattr(network, name, LinearWithLoRA(module, rank, alpha))
                    counts += 1
                except Exception as e:
                    pass
            else:
                replace_linear_with_lora(module, rank, alpha, device, tag=0)
        else:
            if name in module_names:
                try:
                    setattr(network, name, LinearWithLoRA(module, rank, alpha))
                    counts += 1
                except Exception as e:
                    replace_linear_with_lora(module, rank, alpha, device, module_names, tag=0)
            else:
                replace_linear_with_lora(module, rank, alpha, device, module_names, tag=0)
    network.to(device)
    return counts

