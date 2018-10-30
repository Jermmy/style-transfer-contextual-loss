import torch
import torch.nn as nn
import torch.nn.functional as F


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

        print(featureI.shape)

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

        print(dist.shape)

        raw_dist = (1. - dist) / 2.

        relative_dist = self.calc_relative_distances(raw_dist)

        CX = self.calc_CX(relative_dist)

        CX = CX.max(dim=2)[0].max(dim=1)[0]
        CX = CX.mean(1)
        CX = -torch.log(CX)
        CX = torch.mean(CX)
        return CX


def init_params(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Down(nn.Module):

    def __init__(self, size, in_channels, out_channels):
        super(Down, self).__init__()
        self.size = size
        self.features = [nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
                         nn.LayerNorm([out_channels, size, size]),
                         nn.LeakyReLU(0.2),
                         nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
                         nn.LayerNorm([out_channels, size, size]),
                         nn.LeakyReLU(0.2)]
        self.features = nn.Sequential(*self.features)
        self.upsample = nn.Upsample(size=(self.size, self.size), mode='bilinear')

    def forward(self, image, x=None):
        out = self.upsample(image)
        if x is not None:
            out = torch.cat([x, out], dim=1)

        return self.features(out)


class Generator(nn.Module):

    def __init__(self, image_size):
        super(Generator, self).__init__()

        self.image_size = image_size

        self.image_sizes = [256, 128, 64, 32, 16, 8, 4]

        self.in_dims = [259, 515, 515, 515, 515, 515, 3]
        self.out_dims = [256, 256, 512, 512, 512, 512, 512]

        self.rec = nn.Sigmoid()
        self.conv = nn.Conv2d(256, 3, kernel_size=(1, 1))
        self.down1 = Down(self.image_sizes[0], 259, 256)
        self.donw2 = Down(self.image_sizes[1], 515, 256)
        self.down3 = Down(self.image_sizes[2], 515, 512)
        self.down4 = Down(self.image_sizes[3], 515, 512)
        self.down5 = Down(self.image_sizes[4], 515, 512)
        self.down6 = Down(self.image_sizes[5], 515, 512)
        self.down7 = Down(self.image_sizes[6], 3, 512)

        init_params(self.modules())

    def forward(self, x):
        down7 = self.down7(x)
        down7 = F.interpolate(down7, size=(self.image_sizes[-2], self.image_sizes[-2]), mode='bilinear')
        down6 = self.down6(x, down7)
        down6 = F.interpolate(down6, size=(self.image_sizes[-3], self.image_sizes[-3]), mode='bilinear')
        down5 = self.down5(x, down6)
        down5 = F.interpolate(down5, size=(self.image_sizes[-4], self.image_sizes[-4]), mode='bilinear')
        down4 = self.down4(x, down5)
        down4 = F.interpolate(down4, size=(self.image_sizes[-5], self.image_sizes[-5]), mode='bilinear')
        down3 = self.down3(x, down4)
        down3 = F.interpolate(down3, size=(self.image_sizes[-6], self.image_sizes[-6]), mode='bilinear')
        down2 = self.donw2(x, down3)
        down2 = F.interpolate(down2, size=(self.image_sizes[-7], self.image_sizes[-7]), mode='bilinear')
        down1 = self.down1(x, down2)
        return (self.conv(down1) + 1.) / 2.


# class Down(nn.Module):
#
#     def __init__(self, in_channels, out_channels, BN=False, IN=True):
#         super(Down, self).__init__()
#         modules = [nn.LeakyReLU(0.2, inplace=False),
#                   nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
#         if BN:
#             modules.append(nn.BatchNorm2d(out_channels))
#         if IN:
#             modules.append(nn.InstanceNorm2d(out_channels))
#
#         self.feature = nn.Sequential(*modules)
#         init_params(self.feature)
#
#     def forward(self, x):
#         return self.feature(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, BN=False, IN=True, dropout=True):
        super(Up, self).__init__()
        modules = [nn.LeakyReLU(0.2, inplace=False),
                  nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if BN:
            modules.append(nn.BatchNorm2d(out_channels))
        if IN:
            modules.append(nn.InstanceNorm2d(out_channels))
        if dropout:
            modules.append(nn.Dropout(0.5))

        self.feature = nn.Sequential(*modules)

    def forward(self, c1, c2=None):
        c1 = self.feature(c1)
        if c2 is not None:
            return torch.cat([c1, c2], dim=1)
        else:
            return c1


# class Generator(nn.Module):
#
#     def __init__(self, out_channels=64):
#         super(Generator, self).__init__()
#         # 256 x 256 x 6
#         self.down1 = nn.Conv2d(6, out_channels, kernel_size=4, stride=2, padding=1)
#         # 128 x 128 x 64
#         self.down2 = Down(out_channels, out_channels * 2)
#         # 64 x 64 x 128
#         self.down3 = Down(out_channels * 2, out_channels * 2)
#         # 32 x 32 x 256
#         self.down4 = Down(out_channels * 2, out_channels * 2)
#         # 16 x 16 x 512
#         self.down5 = Down(out_channels * 2, out_channels * 2)
#         # 8 x 8 x 512
#         self.down6 = Down(out_channels * 2, out_channels * 2)
#         # 4 x 4 x 512
#         self.down7 = Down(out_channels * 2, out_channels * 2)
#         # 2 x 2 x 512
#         self.down8 = Down(out_channels * 2, out_channels * 2, IN=False)
#
#         # 1 x 1 x 512
#         self.up1 = Up(out_channels * 2, out_channels * 2)
#         # 2 x 2 x (512 + 512)
#         self.up2 = Up(out_channels * 2 * 2, out_channels * 2)
#         # 4 x 4 x (512 + 512)
#         self.up3 = Up(out_channels * 2 * 2, out_channels * 2)
#         # 8 x 8 x (512 + 512)
#         self.up4 = Up(out_channels * 2 * 2, out_channels * 2, dropout=False)
#         # 16 x 16 x (512 + 512)
#         self.up5 = Up(out_channels * 2 * 2, out_channels * 2, dropout=False)
#         # 32 x 32 x (256 + 256)
#         self.up6 = Up(out_channels * 2 * 2, out_channels * 2, dropout=False)
#         # 64 x 64 x (128 + 128)
#         self.up7 = Up(out_channels * 2 * 2, out_channels, dropout=False)
#         # 128 x 128 x (64 + 64)
#         self.up8 = Up(out_channels * 2, 3, IN=False, dropout=False)
#         # 256 x 256 x 3
#         self.rec = nn.Sigmoid()
#
#     def forward(self, s, t):
#         '''
#         :param dImage: degraded image
#         :param wImage: wrap guidance image
#         :return:
#         '''
#         x = torch.cat([s, t], dim=1)
#         down1 = self.down1(x)
#         down2 = self.down2(down1)
#         down3 = self.down3(down2)
#         down4 = self.down4(down3)
#         down5 = self.down5(down4)
#         down6 = self.down6(down5)
#         down7 = self.down7(down6)
#         down8 = self.down8(down7)
#
#         up1 = self.up1(down8, down7)
#         up2 = self.up2(up1, down6)
#         up3 = self.up3(up2, down5)
#         up4 = self.up4(up3, down4)
#         up5 = self.up5(up4, down3)
#         up6 = self.up6(up5, down2)
#         up7 = self.up7(up6, down1)
#         up8 = self.up8(up7)
#         rec = self.rec(up8)
#         return rec
