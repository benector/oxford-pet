import torch
import torch.nn as nn
from .base_model import BaseModel
import torch.nn.functional as F  # ✅ importa o módulo funcional


# ===============================
# Atenção por Canal
# ===============================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        y = torch.mean(x, dim=(2, 3))  # Global Average Pooling
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ===============================
# Atenção Espacial
# ===============================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = torch.sigmoid(self.conv(y))
        return x * y



# ===============================
# Atenção Combinada
# ===============================
class ChannelSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel = ChannelAttention(in_channels)
        self.spatial = SpatialAttention()
    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


# ===============================
# Depthwise Separable Convolution
# ===============================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = dilation if kernel_size == 3 else 0
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# ===============================
# BN + ReLU + Conv (normal ou separável)
# ===============================
class BN_ReLU_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, use_separable=False):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
#        self.dropout2d = nn.Dropout2d(p)

        if use_separable and kernel_size == 3:
            self.conv = DepthwiseSeparableConv(
                in_channels, out_channels,
                kernel_size=3, stride=stride, dilation=dilation
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride,
                padding=dilation if kernel_size == 3 else 0,
                dilation=dilation,
                bias=False
            )

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
 #       x = self.dropout2d(x)
        return x


# ===============================
# Dense Block
# ===============================
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, use_separable=False, dilation=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    # Bottleneck 1x1
                    BN_ReLU_Conv(in_channels, 4 * growth_rate, kernel_size=1, use_separable=use_separable),
                    # 3x3 conv (separável/dilatada se configurado)
                    BN_ReLU_Conv(4 * growth_rate, growth_rate, kernel_size=3,
                                 dilation=dilation, use_separable=use_separable)
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


# ===============================
# Transition Layer
# ===============================
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, use_separable=False, p=0.0):
        super().__init__()
        out_channels = in_channels // 2
        self.bn_relu_conv = BN_ReLU_Conv(in_channels, out_channels, kernel_size=1, use_separable=use_separable)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2d = nn.Dropout2d(p)

    def forward(self, x):
        x = self.bn_relu_conv(x)
        x = self.dropout2d(x)
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
        dropout_p = config["model"]["dropout_p"]
        dropout_dblock = config["model"]["dropout_dblock"]
        dropout_transition = config["model"]["dropout_transition"]

        # novas opções vindas do config.json
        self.use_separable = config['model'].get('use_separable', False)
        self.dilation = config['model'].get('dilation', 1)

        #atençao opcional
        self.use_channel_attention = config['model'].get('channel_attention', False)
        self.use_spatial_attention = config['model'].get('spatial_attention', False)
        self.trans_attention = config['model'].get('trans_attention', False)
        self.block_attention = config['model'].get('block_attention', False)
        self.global_attention = config['model'].get('global_attention', False)



        print("\n===== ESTADO DAS ATENÇÕES =====")
        print(f"→ Channel Attention:   {'✅ ATIVADA' if self.use_channel_attention else '❌ DESATIVADA'}")
        print(f"→ Spatial Attention:   {'✅ ATIVADA' if self.use_spatial_attention else '❌ DESATIVADA'}")
        print(f"→ Block Attention:     {'✅ ATIVADA' if self.block_attention else '❌ DESATIVADA'}")
        print(f"→ Transition Attention:{'✅ ATIVADA' if self.trans_attention else '❌ DESATIVADA'}")
        print(f"→ Global Attention:    {'✅ ATIVADA' if self.global_attention else '❌ DESATIVADA'}")
        print("================================\n")


        # camada inicial
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # blocos densos + transições
        num_layers = [6, 12, 24, 16]  # DenseNet121
        in_channels = 64

        self.blocks = nn.ModuleList()
        for i, layers in enumerate(num_layers):
        # Cria DenseBlock
            block = DenseBlock(in_channels, growth_rate, layers,
                        use_separable=self.use_separable,
                        dilation=self.dilation)
            self.blocks.append(block)
            in_channels = block.out_channels

        # 🔹 Atenção após o DenseBlock (se configurado)
            if self.block_attention:
                if self.use_channel_attention and self.use_spatial_attention:
                      self.blocks.append(ChannelSpatialAttention(in_channels))
#                     self.blocks.append(ChannelAttention(in_channels))
#                     self.blocks.append(SpatialAttention())
                elif self.use_channel_attention:
                    self.blocks.append(ChannelAttention(in_channels))
                elif self.use_spatial_attention:
                    self.blocks.append(SpatialAttention())

        # 🔹 Cria camada de transição (exceto após o último bloco)
            if i != len(num_layers) - 1:
                trans = TransitionLayer(in_channels,
                                    use_separable=self.use_separable,
                                    p=dropout_transition)
                self.blocks.append(trans)
                in_channels = in_channels // 2

            # 🔹 Atenção após a transição (se configurado)
                if self.trans_attention:
                    if self.use_channel_attention and self.use_spatial_attention:
                        self.blocks.append(ChannelSpatialAttention(in_channels))
#                        self.blocks.append(ChannelAttention(in_channels))
#                        self.blocks.append(SpatialAttention())
                    elif self.use_channel_attention:
                        self.blocks.append(ChannelAttention(in_channels))
                    elif self.use_spatial_attention:
                        self.blocks.append(SpatialAttention())

        # camada final
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(in_channels, num_classes)
        self.global_att = ChannelAttention(in_channels) if self.global_attention else None


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        for block in self.blocks:
            x = block(x)

        x = self.bn(x)
        x = self.relu(x)
        if self.global_att is not None:
            x = self.global_att(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
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
