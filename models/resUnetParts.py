import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualDoubleConv, self).__init__()
        self.resconv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        residual = self.resconv(x)  # Save the input as the residual connection
        x = self.double_conv(x)
        x = x + residual  # Add the residual connection
        return x


class ResDown(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.maxpool_conv = nn.Sequential(
            # Max-pooling untuk downsampling
            nn.MaxPool2d(2),
            # Dua lapisan konvolusi berurutan
            ResidualDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ResUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if bilinear:
            # Upsampling menggunakan bilinear interpolation
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResidualDoubleConv(in_channels, out_channels // 2)
        else:
            # Upsampling menggunakan konvolusi transpose
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResidualDoubleConv(in_channels, out_channels)

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


if __name__ == '__main__':
    x = torch.rand(1,3,10,10)
    resblock = ResidualDoubleConv(3,10)
    print(resblock(x).size())
    downcon = ResDown(10,20)
    print(downcon(resblock(x)).size())

