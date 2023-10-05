import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Encoding
        x1 = self.encoder[0](x)
        x2 = self.encoder[2](x1)
        x3 = self.encoder[4](x2)
        x4 = self.encoder[6](x3)

        # Decoding
        x = self.decoder[0](x4)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder[1](x)
        x = self.decoder[2](x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder[3](x)
        x = self.decoder[4](x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder[5](x)
        x = self.decoder[6](x)
        return x