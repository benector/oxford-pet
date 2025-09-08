import torch
import torch.nn as nn
from .base_model import BaseModel



# ===============================
# DenseNet building blocks
# ===============================
class BN_ReLU_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False
        )
        
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    BN_ReLU_Conv(in_channels, 4 * growth_rate),   # bottleneck
                    BN_ReLU_Conv(4 * growth_rate, growth_rate, kernel_size=3)
                )
            )
            in_channels += growth_rate
        self.out_channels = in_channels

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            y = layer(torch.cat(features, 1))
            features.append(y)
        return torch.cat(features, 1)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels // 2
        self.bn_relu_conv = BN_ReLU_Conv(in_channels, out_channels)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.bn_relu_conv(x)
        x = self.pool(x)
        return x


# ===============================
# DenseNet121
# ===============================
class DenseNet121(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        input_channels = config['model']['in_channels']
        num_classes = config['model']['num_classes']
        growth_rate = config['model']['growth_rate'] 

        # camada inicial
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # blocos densos + transições
        num_layers = [6, 12, 24, 16]  # DenseNet121
        in_channels = 64

        self.blocks = nn.ModuleList()
        for i, layers in enumerate(num_layers):
            block = DenseBlock(in_channels, growth_rate, layers)
            self.blocks.append(block)
            in_channels = block.out_channels
            if i != len(num_layers) - 1:
                trans = TransitionLayer(in_channels)
                self.blocks.append(trans)
                in_channels = in_channels // 2

        # camada final
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        for block in self.blocks:
            x = block(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

# ===============================
# Factory
# ===============================
def get_model(config):
    model_name = config['model']['name']
    if model_name == "DenseNet121":
        return DenseNet121(config)
    else:
        raise ValueError(f"Modelo {model_name} não encontrado")
