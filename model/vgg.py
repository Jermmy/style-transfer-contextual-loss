from torchvision.models import VGG, vgg19
import torch.nn as nn


def conv2d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True))


def conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class VGG19(nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()
        self.conv1_1 = nn.Sequential(conv(3, 64), nn.ReLU())
        self.conv1_2 = nn.Sequential(conv(64, 64), nn.ReLU())
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.conv2_1 = nn.Sequential(conv(64, 128), nn.ReLU())
        self.conv2_2 = nn.Sequential(conv(128, 128), nn.ReLU())
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.conv3_1 = nn.Sequential(conv(128, 256), nn.ReLU())
        self.conv3_2 = nn.Sequential(conv(256, 256), nn.ReLU())
        self.conv3_3 = nn.Sequential(conv(256, 256), nn.ReLU())
        self.conv3_4 = nn.Sequential(conv(256, 256), nn.ReLU())
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.conv4_1 = nn.Sequential(conv(256, 512), nn.ReLU())
        self.conv4_2 = nn.Sequential(conv(512, 512), nn.ReLU())

    def forward(self, input_images):

        feature = {}
        feature['conv1_1'] = self.conv1_1(input_images)
        feature['conv1_2'] = self.conv1_2(feature['conv1_1'])
        feature['pool1'] = self.pool1(feature['conv1_2'])
        feature['conv2_1'] = self.conv2_1(feature['pool1'])
        feature['conv2_2'] = self.conv2_2(feature['conv2_1'])
        feature['pool2'] = self.pool2(feature['conv2_2'])
        feature['conv3_1'] = self.conv3_1(feature['pool2'])
        feature['conv3_2'] = self.conv3_2(feature['conv3_1'])
        feature['conv3_3'] = self.conv3_3(feature['conv3_2'])
        feature['conv3_4'] = self.conv3_4(feature['conv3_3'])
        feature['pool3'] = self.pool3(feature['conv3_4'])
        feature['conv4_1'] = self.conv4_1(feature['pool3'])
        feature['conv4_2'] = self.conv4_2(feature['conv4_1'])

        return feature
