import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

class CXLoss(nn.Module):

    def __init__(self, sigma=0.1, b=1.0, similarity="consine"):
        super(CXLoss, self).__init__()
        self.similarity = similarity
        self.sigma = sigma
        self.b = b

    def center_by_T(self, featureI, featureT):
        # Calculate mean channel vector for feature map.
        meanT = featureT.mean(0, keepdim=True).mean(1, keepdim=True).mean(2, keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=3)
        norms = features.norm(p=2, dim=3, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, features):
        N, H, W, C = features.shape
        assert N == 1
        P = H * W
        # NHWC --> 1x1xHWxC --> HWxCx1x1
        patches = features.reshape(shape=(1, 1, P, C)).permute((2, 3, 0, 1))
        return patches

    def calc_relative_distances(self, raw_dist, axis=3):
        epsilon = 1e-5
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist

    def calc_CX(self, dist, axis=3):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W.div(W_sum)

    def forward(self, featureT, featureI):
        '''
        :param featureT: target
        :param featureI: inference
        :return:
        '''
        # NCHW --> NHWC
        featureI = featureI.permute((0, 2, 3, 1))
        featureT = featureT.permute((0, 2, 3, 1))

        featureI, featureT = self.center_by_T(featureI, featureT)

        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        dist = []
        N = featureT.size()[0]
        for i in range(N):
            featureT_i = featureT[i, :, :, :].unsqueeze_(0)
            # NHWC --> NCHW
            featureI_i = featureI[i, :, :, :].unsqueeze_(0).permute((0, 3, 1, 2))
            featureT_patch = self.patch_decomposition(featureT_i)
            # Calculate cosine similarity
            dist_i = F.conv2d(featureI_i, featureT_patch)
            dist_i_1HWC = dist_i.permute((0, 2, 3, 1))
            dist.append(dist_i_1HWC)

        # NHWC
        dist = torch.cat(dist, dim=0)

        raw_dist = (1. - dist) / 2.

        relative_dist = self.calc_relative_distances(raw_dist)

        CX = self.calc_CX(relative_dist)

        CX = CX.max(dim=2)[0].max(dim=1)[0]
        CX = CX.mean(1)
        CX = -torch.log(CX)
        CX = torch.mean(CX)
        return CX


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.image_sizes = [256, 128, 64, 32, 16, 8, 4]

        self.in_dims = [259, 515, 515, 515, 515, 515, 3]
        self.out_dims = [256, 256, 512, 512, 512, 512, 512]

        self.features = []
        self.transforms = []

        for i in range(len(self.image_sizes)):
            if i == 0:
                self.features.append(nn.Sequential(
                    nn.Conv2d(in_channels=self.in_dims[i], out_channels=self.out_dims[i], kernel_size=3),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(in_channels=self.out_dims[i], out_channels=self.out_dims[i], kernel_size=3),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(in_channels=self.out_dims[i], out_channels=3, kernel_size=1)
                ))
            else:
                self.features.append(nn.Sequential(
                    nn.Conv2d(in_channels=self.in_dims[i], out_channels=self.out_dims[i], kernel_size=3),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(in_channels=self.out_dims[i], out_channels=self.out_dims[i], kernel_size=3),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            self.transforms.append(transforms.Resize((self.image_sizes[i], self.image_sizes[i])))

    def forward(self, x):
        for i in range(len(self.image_sizes)-1, -1, -1):
            if i == len(self.image_sizes) - 1:
                resize_x = self.transforms[i](x)
                conv_x = self.features[i](resize_x)
            else:
                conv_x = self.transforms[i](conv_x)
                resize_x = self.transforms[i](x)
                conv_x = torch.cat([conv_x, resize_x], dim=1)
                conv_x = self.features[i](conv_x)

        conv_x = (conv_x + 1.) / 2. * 255.
        return conv_x






