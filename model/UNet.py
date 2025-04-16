import torch 
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, input):
        return self.block(input)
    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, stride=2)
    
    def forward(self, input):
        skip = self.down(input)
        out = self.pool(skip)
        return out, skip

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Here the out_channels is half that of the in_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ConvBlock(2 * out_channels, out_channels)
    
    def forward(self, input, skip):
        # the skip image is currently larger than the input after upsampled
        input = self.up(input)

        # TODO Investigate croping aspect of paper
        """
        crop = (skip.shape[-1] - input.shape[-1]) // 2 
        if skip.shape[-1] % 2 == 1:
            skip = skip[:, :, crop:-crop-1, crop:-crop-1]
        else:
            skip = skip[:, :, crop:-crop, crop:-crop]
        """
        combine = torch.cat([input, skip], dim = 1)

        return self.block(combine)



class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)

        self.convblock = ConvBlock(256, 512)

        self.up1 = UpSample(512, 256)
        self.up2 = UpSample(256, 128)
        self.up3 = UpSample(128, 64)

        self.output = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, input):
        out, skip1 = self.down1(input)
        out, skip2 = self.down2(out)
        out, skip3 = self.down3(out)

        out = self.convblock(out)

        out = self.up1(out, skip3)
        out = self.up2(out, skip2)
        out = self.up3(out, skip1)

        return self.output(out)
