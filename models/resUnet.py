import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualDoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
        )

    def forward(self, x):
        residual = x  # Save the input as the residual connection
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


class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, init_filter=64, bilinear=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Definisi komponen U-Net
        self.inputTensor = (ResidualDoubleConv(n_channels, init_filter))
        self.down1 = (ResDown(init_filter, init_filter*2))
        self.down2 = (ResDown(init_filter*2, init_filter*4))
        self.down3 = (ResDown(init_filter*4, init_filter*8))
        factor = 2 if bilinear else 1
        self.down4 = (ResDown(init_filter*8, init_filter*16 // factor))
        self.up1 = (ResUp(init_filter*16, init_filter*8 // factor, bilinear))
        self.up2 = (ResUp(init_filter*8, init_filter*4 // factor, bilinear))
        self.up3 = (ResUp(init_filter*4, init_filter*2 // factor, bilinear))
        self.up4 = (ResUp(init_filter*2, init_filter, bilinear))
        self.outputTensor = (OutConv(init_filter, n_classes))

    def forward(self, x):
        x1 = self.inputTensor(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outputTensor(x)
        return logits

    def use_checkpointing(self):
        # Menerapkan checkpointing di beberapa lapisan.
        self.inputTensor = torch.utils.checkpoint(self.inputTensor)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outputTensor = torch.utils.checkpoint(self.outputTensor)
