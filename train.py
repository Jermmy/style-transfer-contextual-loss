from model.generator import Generator, CXLoss
from model.vgg import VGG19
from dataloader.dataset import TrainDataset, TestDataset
from model.vgg import VGG19
import torch
import torch.optim as optim
import torchvision.transforms as transforms

import argparse
import collections
import os
from os.path import exists, join

def main(config):

    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor()
    ])

    train_dataset = TrainDataset(train_dir=config.train_dir, style_image=config.style_image, transforms=transform)
    test_dataset = TestDataset(test_dir=config.test_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,
                                               shuffle=True, num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    vgg19 = VGG19().to(device)
    # Loading pretrained model
    pretrained_dict = torch.load(config.vgg)
    temp_dict = collections.OrderedDict()
    vgg19_dict = vgg19.state_dict()
    for k, v in pretrained_dict.items():
        # Update conv3_2
        if k.replace('features', 'conv3_2') in vgg19_dict:
            temp_dict[k.replace('features', 'conv3_2')] = v
        # Update conv4_2
        elif 'conv4_2.' + str(int(k.split('.')[1])-14) + '.' + k.split('.')[2] in vgg19_dict:
            temp_dict['conv4_2.' + str(int(k.split('.')[1])-14) + '.' + k.split('.')[2]] = v
    vgg19_dict.update(temp_dict)
    vgg19.load_state_dict(vgg19_dict)
    # Fix parameters
    vgg19.eval()

    generator = Generator(image_size=config.image_size).to(device)
    cxloss = CXLoss().to(device)

    optimizer = optim.Adam(params=generator.parameters(), lr=config.lr)

    if not exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)

    if not exists(config.result_path):
        os.makedirs(config.result_path)

    for epoch in range(1, config.epochs + 1):

        for i, data in enumerate(train_loader):
            content = data['content'].to(device).float()
            style = data['style'].to(device).float()

            optimizer.zero_grad()

            _, vgg_content = vgg19(content)
            vgg_style, _ = vgg19(style)
            fake = generator(content)
            print(fake.shape)
            fake_style, fake_content = vgg19(fake)

            cx_style_loss = config.lambda_style * cxloss(vgg_style, fake_style)
            cx_content_loss = config.lambda_content * cxloss(vgg_content, fake_content)

            loss = cx_style_loss + cx_content_loss

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("Epoch: %d/%d | Step: %d/%d | Style loss: %f | Content loss: %f | Loss: %f" %
                      (epoch, config.epochs+1, i, len(train_loader), cx_style_loss.item(), cx_content_loss.item(), loss.item()))

            # if i % 100 == 0:

            if (i + 1) % 500 == 0:
                torch.save(generator.state_dict(), join(config.ckpt_path, 'epoch-%d.pkl' % epoch))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--train_dir', type=str, default='single/train')
    parser.add_argument('--style_image', type=str, default='single/4.jpg')
    parser.add_argument('--test_dir', type=str, default='single/test')
    parser.add_argument('--vgg', type=str, default='vgg19-dcbb9e9d.pth')
    parser.add_argument('--ckpt_path', type=str, default='ckpts/')
    parser.add_argument('--result_path', type=str, default='results/')
    parser.add_argument('--lambda_style', type=float, default=1.0)
    parser.add_argument('--lambda_content', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)

    config = parser.parse_args()

    main(config)