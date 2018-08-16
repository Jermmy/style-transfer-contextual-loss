import os
from PIL import Image
import torch
import numpy as np
from torch.utils import data
import torchvision.transforms as transforms


class TrainDataset(data.Dataset):

    def __init__(self, train_dir, style_image, transforms=None):
        self.train_images = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith("jpg")]
        self.style_image = Image.open(style_image)
        self.transforms = transforms

        if self.transforms:
            self.style_image = self.transforms(self.style_image)

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        content_image = Image.open(self.train_images[idx])

        if content_image.mode == "L":
            content_image = content_image.convert(mode="RGB")

        sample = {'content': content_image, 'style': self.style_image}
        if self.transforms:
            sample['content'] = self.transforms(sample['content'])

        return sample


class TestDataset(data.Dataset):

    def __init__(self, test_dir, transforms=None):
        self.test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith("jpg")]
        self.transforms = transforms

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        image = Image.open(self.test_images[idx])
        image = image.convert(mode="RGB")
        if self.transforms:
            image = self.transforms(image)

        sample = {'content': image}
        return sample
