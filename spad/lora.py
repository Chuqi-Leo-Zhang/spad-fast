import torch
import torch.nn as nn

class LoRALinear(nn.Module):

    def __init__(self, base_layer, r=4, alpha=1.0):
        super().__init__()
        self.base = base_layer
        in_dim = base_layer.in_features
        out_dim = base_layer.out_features

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)

        # nn.init.zeros_(self.lora_A.weight)
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x, use_lora=True):
        if use_lora:
            return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling
        else:
            return self.base(x)
    
    @property
    def weight(self):
        # makes optimizer ignore base.weight correctly
        return self.base.weight
    
    @property
    def bias(self):
        return self.base.bias
    
class LoRAConv2d(nn.Module):

    def __init__(self, base_conv, r=4, alpha=1.0):
        super().__init__()
        assert isinstance(base_conv, nn.Conv2d)
        self.conv = base_conv                      # frozen base conv
        self.r = r
        self.scaling = alpha / r

        in_c = base_conv.in_channels
        out_c = base_conv.out_channels

        # LoRA A: reduce channels
        self.lora_down = nn.Conv2d(
            in_c, r, kernel_size=1, stride=1, padding=0, bias=False
        )
        # LoRA B: expand channels
        self.lora_up = nn.Conv2d(
            r, out_c, kernel_size=1, stride=1, padding=0, bias=False
        )

        # initialize delta to 0 (important!)
        nn.init.zeros_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x, use_lora=False):
        if use_lora:
            return self.conv(x) + self.lora_up(self.lora_down(x)) * self.scaling
        else:
            return self.conv(x)

    @property
    def weight(self):
        # makes optimizer ignore conv.weight correctly
        return self.conv.weight
