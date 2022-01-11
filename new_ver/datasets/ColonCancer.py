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
from skimage import io, color
import scipy.io
import re
import numpy as np
import math

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
                 data_dir="datasets/data/colon_cancer/",
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
                    patch = self.crop_patch(x, y, img)
                    bag.append(patch)

            try:
                np.array(bag).reshape(-1, 27, 27, 3)
            except:
                pass

            self.bags.append(np.array(bag).reshape(-1, 27, 27, 3))

            label = 1 if "epithelial" in img_path else 0
            self.labels.append(label)

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

    def crop_patch(self, x, y, img, padding=None):
        x, y = round(x), round(y)
        patch = None

        if padding:
            padded_img = self.add_padding(img, padding)

            offset = math.ceil(padding / 2)
            x += offset
            y += offset

            x0, x1 = x - offset, x + offset - 1
            y0, y1 = y - offset, y + offset - 1

            try:
                patch = padded_img[x0:x1, y0:y1]
            except IndexError:
                pass
            finally:
                return patch
        else:
            x0, x1 = self.get_constraints(x, 0, img.shape[0], 13, 14)
            y0, y1 = self.get_constraints(y, 0, img.shape[1], 13, 14)
            try:
                patch = img[x0:x1, y0:y1]
            except IndexError:
                pass
            finally:
                return patch

    @staticmethod
    def add_padding(img, padding):
        height, width, channels = img.shape

        new_height = height + padding
        new_width = width + padding

        result = np.full((new_height, new_width, channels), (0, 0, 0), dtype=np.uint8)

        x_center = math.ceil((new_width - width) / 2)
        y_center = math.ceil((new_height - height) / 2)

        result[y_center:y_center + height, x_center:x_center + width] = img

        return result

    @staticmethod
    def get_constraints(a, min_a, max_a, offset_l, offset_r):
        a0, a1 = None, None

        if a - offset_l >= min_a and a + offset_r <= max_a:
            a0 = a - offset_l
            a1 = a + offset_r
        elif a - offset_l >= min_a and a + offset_r >= max_a:
            a0 = max_a - (offset_l + offset_r)
            a1 = max_a
        else:
            a0 = 0
            a1 = offset_l + offset_r

        return a0, a1


if __name__ == "__main__":
    # preview()
    pass
