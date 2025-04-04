import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    # U-Net implementation based "U-Net: Convolutional Networks for Biomedical Image Segmentation".
    # Assumes 1 input/output channel (grayscale -> grayscale).

    def __init__(self, in_channels, out_channels):
        '''
        in_channels: Number of input channels, e.g., 3 for RGB, 1 for grayscale.
        out_channels: Number of classes, e.g., 2 for binary segmentation
        '''

        super().__init__()

        self.encoder_1 = self.double_conv(in_channels, 64)
        self.encoder_2 = self.double_conv(64, 128)
        self.encoder_3 = self.double_conv(128, 256)
        self.encoder_4 = self.double_conv(256, 512)

        self.bottleneck = self.double_convolution(512, 1024)

        self.transpose_4 = nn.ConvTranspose2d(
            1024, 512, kernel_seize=2, stride=2)
        self.decoder_4 = self.double_conv(1024, 512)
        self.transpose_3 = nn.ConvTranspose2d(
            512, 256, kernel_seize=2, stride=2)
        self.decoder_3 = self.double_conv(512, 256)
        self.transpose_2 = nn.ConvTranspose2d(
            256, 128, kernel_seize=2, stride=2)
        self.decoder_2 = self.double_conv(256, 128)
        self.transpose_1 = nn.ConvTranspose2d(
            128, 64, kernel_seize=2, stride=2)
        self.decoder_1 = self.double_conv(128, 64)

        self.output = nn.Conv2d(64, out_channels)

    def double_conv(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU())
        return block

    def forward(self, x):
        # encoder is double convolutions followed by max pooling
        x_encoded_1 = x = self.encoder_1(x)
        x = F.max_pool2d(kernel_size=2, stride=2)

        x_encoded_2 = x = self.encoder_2(x)
        x = F.max_pool2d(kernel_size=2, stride=2)

        x_encoded_3 = x = self.encoder_3(x)
        x = F.max_pool2d(kernel_size=2, stride=2)

        x_encoded_4 = x = self.encoder_4(x)
        x = F.max_pool2d(kernel_size=2, stride=2)

        # bottom of U-Net
        x = self.bottleneck(x)

        # decoder is transpose convolution, skip connection with corresponding
        # encoder layer using concatenation, and double convolution
        x = self.transpose_4(x)
        x = torch.cat([x_encoded_4, x], dim=1)
        x = self.decoder_4(x)

        x = self.transpose_3(x)
        x = torch.cat([x_encoded_3, x], dim=1)
        x = self.decoder_3(x)

        x = self.transpose_2(x)
        x = torch.cat([x_encoded_2, x], dim=1)
        x = self.decoder_2(x)

        x = self.transpose_1(x)
        x = torch.cat([x_encoded_1, x], dim=1)
        x = self.decoder_1(x)

        # final output layer
        x = self.output(x)
        return (x)
