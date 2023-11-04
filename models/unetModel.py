from .unetParts import *

# Kode ini mendefinisikan model U-Net utama yang menggabungkan semua komponen.


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, init_filter=64, bilinear=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Definisi komponen U-Net
        self.inputTensor = (DoubleConv(n_channels, init_filter))
        self.down1 = (Down(init_filter, init_filter*2))
        self.down2 = (Down(init_filter*2, init_filter*4))
        self.down3 = (Down(init_filter*4, init_filter*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(init_filter*8, init_filter*16 // factor))
        self.up1 = (Up(init_filter*16, init_filter*8 // factor, bilinear))
        self.up2 = (Up(init_filter*8, init_filter*4 // factor, bilinear))
        self.up3 = (Up(init_filter*4, init_filter*2 // factor, bilinear))
        self.up4 = (Up(init_filter*2, init_filter, bilinear))
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
