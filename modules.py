import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        # Encoder
        self.e11 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.e12 = DoubleConv(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.e21 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Change stride to 2

        self.e31 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.e41 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.e51 = DoubleConv(512, 1024)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # Change stride to 2
        self.d11 = DoubleConv(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # Change stride to 2
        self.d21 = DoubleConv(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # Change stride to 2
        self.d31 = DoubleConv(256, 128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Change stride to 2
        self.d41 = DoubleConv(128, 64)

        # Output layer
        self.outconv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = F.relu(self.e11(x))
        xe12 = self.e12(xe11)
        xp1 = self.pool1(xe12)


        xe21 = self.e21(xp1)
        xp2 = self.pool2(xe21)


        xe31 = self.e31(xp2)
        xp3 = self.pool3(xe31)


        xe41 = self.e41(xp3)
        xp4 = self.pool4(xe41)

        xe51 = self.e51(xp4)

        # Decoder
        xu1 = self.upconv1(xe51)
        xu11 = torch.cat([xu1, xe41], dim=1)
        xd11 = self.d11(xu11)

        xu2 = self.upconv2(xd11)
        xu22 = torch.cat([xu2, xe31], dim=1)
        xd21 = self.d21(xu22)

        xu3 = self.upconv3(xd21)
        xu33 = torch.cat([xu3, xe21], dim=1)
        xd31 = self.d31(xu33)

        xu4 = self.upconv4(xd31)
        xu44 = torch.cat([xu4, xe11], dim=1)
        xd41 = self.d41(xu44)

        # Output layer
        out = self.outconv(xd41)

        return out

    def dice_coefficient(self, predicted, target, smooth=1e-5):
        intersection = (predicted * target).sum()
        union = predicted.sum() + target.sum()
        return (2.0 * intersection + smooth) / (union + smooth)

    def categorical_cross_entropy_loss(self, predicted, target):
        loss = F.cross_entropy(predicted, target)
        return loss
