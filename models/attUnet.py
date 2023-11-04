import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        spatial_gate = torch.sigmoid(self.conv(x))
        return x * spatial_gate


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels)
        )

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        channel_gate = self.fc(avg_pool.view(avg_pool.size(0), -1))
        channel_gate = torch.sigmoid(channel_gate).view(
            x.size(0), x.size(1), 1, 1)
        return x * channel_gate


class DoubleConvBlockWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlockWithAttention, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.spatial_attention = SpatialAttention(out_channels)
        self.channel_attention = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.double_conv(x)
        spatial_att = self.spatial_attention(x)
        channel_att = self.channel_attention(x)
        x = x + spatial_att + channel_att
        return x
