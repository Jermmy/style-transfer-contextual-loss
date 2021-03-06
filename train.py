import argparse
import os
from os.path import exists, join

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from dataloader.dataset import TestDataset, TrainDataset
from model.generator import Generator, CXLoss
from model.vgg import VGG19
# from model.CX_distance import CX_loss


def tensor2image(T):
    image = T.detach().cpu().numpy()[0].transpose((1, 2, 0))
    image = image * 255.
    image = np.minimum(np.maximum(image, 0.0), 255.0)

    return image.astype(np.uint8)


style_layer = ['conv3_2', 'conv4_2']
content_layer = ['conv4_2']


def main(config):

    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = TrainDataset(train_dir=config.train_dir, style_image=config.style_image, transforms=transform)
    test_dataset = TestDataset(test_dir=config.test_dir, transforms=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,
                                               shuffle=True, num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    vgg19 = VGG19().to(device)
    # Loading pretrained model
    vgg19.load_model(config.vgg)
    # Fix parameters
    vgg19.eval()

    generator = Generator(image_size=config.image_size).to(device)
    # generator = Generator().to(device)
    CX_loss = CXLoss(sigma=0.5).to(device)

    if config.load_model:
        generator.load_state_dict(torch.load(config.load_model))

    optimizer = optim.Adam(params=generator.parameters(), lr=config.lr)

    if not exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)

    if not exists(config.result_path):
        os.makedirs(config.result_path)

    for epoch in range(1 + config.start_idx, config.epochs + 1):

        for i, data in enumerate(train_loader):
            source = data['source'].to(device).float()
            style = data['style'].to(device).float()

            optimizer.zero_grad()

            vgg_source = vgg19(source)
            vgg_style = vgg19(style)

            # fake = generator(source, style)
            fake = generator(source)
            vgg_fake = vgg19(fake)

            cx_style_loss = 0
            for s in style_layer:
                cx_style_loss += CX_loss(vgg_style[s], vgg_fake[s])
            cx_style_loss *= config.lambda_style

            cx_content_loss = 0
            for s in content_layer:
                cx_content_loss += CX_loss(vgg_source[s], vgg_fake[s])
            cx_content_loss *= config.lambda_content

            loss = cx_style_loss + cx_content_loss

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("Epoch: %d/%d | Step: %d/%d | Style loss: %f | Content loss: %f | Loss: %f" %
                      (epoch, config.epochs, i, len(train_loader), cx_style_loss.item(), cx_content_loss.item(),
                       loss.item()))

            if (i + 1) % 500 == 0:
                torch.save(generator.state_dict(), join(config.ckpt_path, 'epoch-%d.pkl' % epoch))

            if i % 100 == 0:
                plt.subplot(131)
                plt.imshow(tensor2image(source))
                plt.title('source')
                plt.subplot(132)
                plt.imshow(tensor2image(style))
                plt.title('style')
                plt.subplot(133)
                plt.imshow(tensor2image(fake))
                plt.title('fake')
                plt.tight_layout()
                plt.savefig(join(config.result_path, 'epoch-%d-step-%d.png' % (epoch, i)))

        generator.eval()
        result_path = join(config.result_path, 'epoch-%d' % epoch)
        if not exists(result_path):
            os.makedirs(result_path)
        for i, data in enumerate(test_loader):
            source = data['source'].to(device).float()
            # fake = generator(source, style)
            fake = generator(source)
            plt.subplot(131)
            plt.imshow(tensor2image(source))
            plt.title('content')
            plt.subplot(132)
            plt.imshow(tensor2image(style))
            plt.title('style')
            plt.subplot(133)
            plt.imshow(tensor2image(fake))
            plt.title('fake')
            plt.savefig(join(result_path, 'step-%d.png' % (i+1)))
        generator.train()


def test(config):
    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])

    test_dataset = TestDataset(test_dir=config.test_dir, transforms=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    generator = Generator(image_size=config.image_size).to(device)

    if config.load_model:
        generator.load_state_dict(torch.load(config.load_model))

    if not exists(config.result_path):
        os.makedirs(config.result_path)

    generator.eval()
    if not exists(config.result_path):
        os.makedirs(config.result_path)
    for i, data in enumerate(test_loader):
        source = data['source'].to(device).float()
        fake = generator(source)
        plt.subplot(121)
        plt.imshow(tensor2image(source))
        plt.title('source')
        plt.subplot(122)
        plt.imshow(tensor2image(fake))
        plt.title('fake')
        plt.savefig(join(config.result_path, 'step-%d.png' % (i+1)))
    generator.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--train_dir', type=str, default='dataset/train')
    parser.add_argument('--style_image', type=str, default='dataset/4.jpg')
    parser.add_argument('--test_dir', type=str, default='dataset/test')
    parser.add_argument('--vgg', type=str, default='vgg19-dcbb9e9d.pth')
    parser.add_argument('--ckpt_path', type=str, default='ckpts/')
    parser.add_argument('--result_path', type=str, default='results/')
    parser.add_argument('--lambda_style', type=float, default=1.0)
    parser.add_argument('--lambda_content', type=float, default=1.0)
    parser.add_argument('--lambda_rec', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=0)
    # 1 train | 0 test
    parser.add_argument('--train', type=int, default=1)

    config = parser.parse_args()

    if config.train == 1:
        main(config)
    elif config.train == 0:
        test(config)
