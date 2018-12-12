import os
from PIL import Image
from torch.utils import data


class TrainDataset(data.Dataset):

    def __init__(self, train_dir, style_image, transforms=None):
        self.train_images = [os.path.join(train_dir, f)
                             for f in os.listdir(train_dir) if f.endswith("jpg")]
        self.style_image = style_image
        self.transforms = transforms
        #
        # if self.transforms:
        #     self.style_image = self.transforms(self.style_image)

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        source_image = Image.open(self.train_images[idx])
        style_image = Image.open(self.style_image)

        if source_image.mode == "L":
            source_image = source_image.convert(mode="RGB")

        sample = {'source': source_image, 'style': style_image}
        if self.transforms:
            sample['source'] = self.transforms(sample['source'])
            sample['style'] = self.transforms(sample['style'])

        return sample


class TestDataset(data.Dataset):

    def __init__(self, test_dir, transforms=None):
        self.test_images = [os.path.join(test_dir, f)
                            for f in os.listdir(test_dir) if f.endswith("jpg")]
        self.transforms = transforms

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        image = Image.open(self.test_images[idx])
        image = image.convert(mode="RGB")
        if self.transforms:
            image = self.transforms(image)

        sample = {'source': image}
        return sample
