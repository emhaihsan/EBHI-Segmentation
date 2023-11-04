import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat((avg, max_val), dim=1)  # Concatenate along the channel dimension
        x = self.conv(x)
        return torch.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.in_channel = in_channels
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, math.ceil(in_channels // 16), 1),
            nn.Conv2d(math.ceil(in_channels // 16), math.ceil(in_channels // 16), 3, padding=1),
            nn.Conv2d(math.ceil(in_channels // 16), in_channels, 1)
        )

    def forward(self, x):
        avg = torch.mean(x, dim=(-2,-1), keepdim=True)
        max_val, _ = torch.max(x.view(x.size(0), x.size(1), -1), dim=2, keepdim=True)
        max_val = max_val.view(x.size(0), x.size(1), 1, 1)
        outAvg = self.convblock(avg)
        outMax = self.convblock(max_val)
        x = torch.add(outAvg, outMax)
        return torch.sigmoid(x)

class AttentionConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super(AttentionConvBlock, self).__init__()
        self.residual = residual
        self.resconv = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.spatialAtt = SpatialAttention()
        self.channelAtt = ChannelAttention(out_channels)

    def forward(self, x):
        if self.residual:
            residual = self.resconv(x)
        x = self.double_conv(x)
        cam = self.channelAtt(x)
        sam = self.spatialAtt(x)
        camx = torch.mul(x, cam)
        samx = torch.mul(x, sam)
        if self.residual:
            return  camx + samx + residual
        return camx + samx

class AttDown(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.maxpool_conv = nn.Sequential(
            # Max-pooling untuk downsampling
            nn.MaxPool2d(2),
            # Dua lapisan konvolusi berurutan
            AttentionConvBlock(in_channels, out_channels, residual=residual)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class AttUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, residual=False,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if bilinear:
            # Upsampling menggunakan bilinear interpolation
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = AttentionConvBlock(in_channels, out_channels // 2, residual=residual)
        else:
            # Upsampling menggunakan konvolusi transpose
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = AttentionConvBlock(in_channels, out_channels, residual=residual)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Padding untuk mengubah ukuran x1 agar sesuai dengan x2
        x1 = F.pad(x1, [diffX // 2, diffX - diffX //
                   2, diffY // 2, diffY - diffY // 2])

        # Konkatensis x1 dan x2, lalu terapkan dua lapisan konvolusi
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Lapisan konvolusi 1x1 untuk output
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
# Example usage:
if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 128, 128)  # 1 batch, 3 channels, 128x128 image
    model = AttentionConvBlock(3, 64, residual=True)
    output = model(input_tensor)
    print(output.shape)  # Should be (1, 64, 128, 128)
    down = AttDown(64, 128, residual=True)
    output2 = down(output)
    print(output2.shape)
    up = AttUp(128, 64, bilinear=False,residual=True)
    print(up(output2, output).size())