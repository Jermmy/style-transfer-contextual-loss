from model.generator import Generator, CXLoss
from dataloader.dataset import TrainDataset, TestDataset

import torch
import torchvision.transforms as transforms

def main(config):

    print(config)

    device = torch.device('cuda' if torch.cuda.is_avaliable() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor()
    ])

    train_dataset = TrainDataset(train_dir=config.train_dir, style_image=config.style_image, transforms=transform)
    test_dataset = TestDataset(test_dir=config.test_dir)

    

