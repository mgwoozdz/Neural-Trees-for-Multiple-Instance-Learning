"""
Colon Cancer dataset
from each image 27x27 patches are extracted using cells location
"""

import glob
import random
import re

import numpy as np
import scipy.io
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import os
import utils_augmentation
from skimage import io, color
import utils_img

BASIC_TRANSFORMS = transforms.Compose([utils_augmentation.HistoNormalize(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

AUGMENTED_TRANSFORMS = transforms.Compose([utils_augmentation.RandomHEStain(),
                                           utils_augmentation.HistoNormalize(),
                                           utils_augmentation.RandomRotate(),
                                           utils_augmentation.RandomVerticalFlip(),
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
        if data_dir is None:
            data_dir = os.path.join("datasets", "data", "colon_cancer")
        self.create_bags(data_dir)

    def create_bags(self, data_dir):
        # load all data img_paths in sorted order
        img_paths = glob.glob(data_dir + "*bmp")
        locations = glob.glob(data_dir + "*mat")

        img_paths.sort(key=lambda path: int(re.search(r"([0-9]+)", path).group(1)))
        locations.sort(key=lambda path: int(re.search(r"([0-9]+)", path).group(1)))
        locations = [locations[i:i+4] for i in range(0, len(locations), 4)]

        for img_path, location_paths in zip(img_paths, locations):
            img = io.imread(img_path)
            if img.shape[2] == 4:
                img = color.rgba2rgb(img)

            if self.keep_imgs:
                self.imgs.append(img)

            bag = []
            for mat_path in location_paths:
                mat = scipy.io.loadmat(mat_path)
                detection = mat['detection']

                for x, y in detection:
                    patch = utils_img.crop_patch(x, y, img, offset_l=13, offset_r=14)

                    if patch is not None:
                        bag.append(patch)

            try:
                bag = np.array(bag).reshape(-1, 27, 27, 3)

                if bag.shape[0] != 0:
                    self.bags.append(bag)

                    label = 1 if "epithelial" in img_path else 0
                    self.labels.append(label)
            except:
                pass

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
