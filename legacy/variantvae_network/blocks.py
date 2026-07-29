import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, dim=3, in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0,
                 bias=False, padding_mode='replicate', mode='CBR'):
        super().__init__()
        blocks = []
        for m in mode:
            if m == 'C':
                Conv = getattr(nn, f'Conv{dim}d')
                blocks.append(Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=bias, padding_mode=padding_mode))
            elif m == 'T':
                ConvTranspose = getattr(nn, f'ConvTranspose{dim}d')
                blocks.append(ConvTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, output_padding=output_padding, bias=bias))
            elif m == 'B':
                BatchNorm = getattr(nn, f'BatchNorm{dim}d')
                blocks.append(BatchNorm(out_channels))
            elif m == 'I':
                InstanceNorm = getattr(nn, f'InstanceNorm{dim}d')
                blocks.append(InstanceNorm(out_channels))
            elif m == 'R':
                blocks.append(nn.ReLU())
            elif m == 'L':
                blocks.append(nn.LeakyReLU(negative_slope=0.2))
            elif m == 'P':
                blocks.append(nn.PReLU(num_parameters=1, init=0.25))
            elif m == 'E':
                blocks.append(nn.ELU(alpha=1.0))
            else:
                raise NotImplementedError(f'Unknown mode: {m}')
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, in_ch, out_ch, mode='CBL', bottleneck=True, padding=1):
        super().__init__()
        # Set activation based on mode
        if mode[-1] == 'L':
            self.activation = nn.LeakyReLU(0.2)
        elif mode[-1] == 'R':
            self.activation = nn.ReLU()
        elif mode[-1] == 'P':
            self.activation = nn.PReLU()
        else:
            self.activation = nn.ReLU()  # Default

        self.bottleneck = bottleneck
        if bottleneck:
            self.conv1 = ConvBlock(dim, in_ch, out_ch//4, kernel_size=1, padding=0, mode=mode)
            self.conv2 = ConvBlock(dim, out_ch//4, out_ch//4, kernel_size=3, padding=padding, mode=mode)
            self.conv3 = ConvBlock(dim, out_ch//4, out_ch, kernel_size=1, padding=0, mode=mode[:-1])
        else:
            self.conv1 = ConvBlock(dim, in_ch, out_ch, kernel_size=3, padding=padding, mode=mode)
            self.conv2 = ConvBlock(dim, out_ch, out_ch, kernel_size=3, padding=padding, mode=mode[:-1])

        if in_ch != out_ch:
            self.shortcut = ConvBlock(dim, in_ch, out_ch, kernel_size=1, padding=0, bias=False, mode='C')
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.bottleneck:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out = x + residual
        return self.activation(out)

# =========================
# 3D CBAM
# =========================

class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        hidden = max(in_planes // reduction, 1)

        self.mlp = nn.Sequential(
            nn.Conv3d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=3):  # 3 is safer for 3D
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=3):
        super().__init__()
        self.ca = ChannelAttention3D(channels, reduction)
        self.sa = SpatialAttention3D(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
