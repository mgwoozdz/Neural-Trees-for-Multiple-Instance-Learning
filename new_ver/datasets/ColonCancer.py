"""
Colon Cancer dataset
from each image 27x27 patches are extracted using cells location
"""

import random
import torch
import glob
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import utils_augemntation
from skimage import io
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt


BASIC_TRANSFORMS = transforms.Compose([utils_augemntation.HistoNormalize(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

AUGMENTED_TRANSFORMS = transforms.Compose([utils_augemntation.RandomHEStain(),
                                           utils_augemntation.HistoNormalize(),
                                           utils_augemntation.RandomRotate(),
                                           utils_augemntation.RandomVerticalFlip(),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class ColonCancer(data_utils.Dataset):

    def __init__(self,
                 data_dir=None,
                 augment=False,
                 shuffle_bag=True,
                 keep_imgs=False):

        self.bags = []
        self.labels = []
        self.imgs = []

        self.keep_imgs = keep_imgs
        self.shuffle_bag = shuffle_bag
        self.transform = AUGMENTED_TRANSFORMS if augment else BASIC_TRANSFORMS
        self.create_bags(data_dir)

    def create_bags(self, data_dir):
        pass

        # load all data paths in sorted order
        paths = glob.glob(data_dir + "*bmp")
        paths.sort()

        # for path in paths:
        #     img = io.imread(path)
        #
        #     if self.keep_imgs:
        #         self.imgs.append(img)
        #
        #     bag = view_as_blocks(img, block_shape=(32, 32, 3)).reshape(-1, 32, 32, 3)
        #     self.bags.append(bag)
        #
        #     label = 1 if "epithelial" in path else 0
        #     self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        if self.shuffle_bag:
            random.shuffle(bag)

        bag_tensors = []
        for instance in bag:
            bag_tensors.append(self.transform(instance))
        bag = torch.stack(bag_tensors)

        label = self.labels[idx]
        img = self.imgs[idx] if self.keep_imgs else []
        return bag, label, img


if __name__ == "__main__":
    # preview()
    pass
