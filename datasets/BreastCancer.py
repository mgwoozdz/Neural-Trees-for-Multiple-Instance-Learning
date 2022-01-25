"""
Breast Cancer dataset
each image (768x896) is split into 32x32 patches (672 patches per image) in a grid-fashion way
"""
import numpy as np
import numpy.ma as ma
import random
import torch
import glob
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import os
import utils_augmentation
from skimage import io
from skimage.util import view_as_blocks


BASIC_TRANSFORMS = transforms.Compose([utils_augmentation.HistoNormalize(),
                                       transforms.ToTensor()])

AUGMENTED_TRANSFORMS = transforms.Compose([utils_augmentation.RandomHEStain(),
                                           utils_augmentation.HistoNormalize(),
                                           utils_augmentation.RandomRotate(),
                                           utils_augmentation.RandomVerticalFlip(),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor()])


class BreastCancer(data_utils.Dataset):

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
            data_dir = os.path.join("datasets", "data", "breast_cancer")
        self.create_bags(data_dir)

    def create_bags(self, data_dir):

        # load all data paths in sorted order
        paths = glob.glob(data_dir + '/*.tif')
        paths.sort()

        for path in paths:
            img = io.imread(path)

            bag = view_as_blocks(img, block_shape=(32, 32, 3)).reshape(-1, 32, 32, 3)
            bag = self.discard_white_patches(bag)
            self.bags.append(bag)

            label = 1 if "malignant" in path else 0
            self.labels.append(label)

            if self.keep_imgs:
                self.imgs.append(img)

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

    def discard_white_patches(self, bag, discard_ratio=0.75, white_threshold=215):

        discard_threshold = 32 * 32 * 3 * discard_ratio
        idxs = []

        for idx, patch in enumerate(bag):
            if np.greater_equal(patch, white_threshold).sum() < discard_threshold:
                idxs.append(idx)

        return bag[idxs]
