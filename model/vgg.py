from torchvision.models import VGG, vgg19
import torch.nn as nn


def conv2d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True))


def conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


class VGG19(nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()
        self.conv3_2 = nn.Sequential(
            conv(3, 64),
            nn.ReLU(inplace=True),
            conv(64, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            conv(64, 128),
            nn.ReLU(inplace=True),
            conv(128, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            conv(128, 256),
            nn.ReLU(inplace=True),
            conv(256, 256),
            nn.ReLU(inplace=True)
        )
        self.conv4_2 = nn.Sequential(
            conv(256, 256),
            nn.ReLU(inplace=True),
            conv(256, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            conv(256, 512),
            nn.ReLU(inplace=True),
            conv(512, 512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv3_2 = self.conv3_2(x)
        conv4_2 = self.conv4_2(conv3_2)
        return conv3_2, conv4_2

