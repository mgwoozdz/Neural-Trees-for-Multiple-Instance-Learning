"""
Breast Cancer dataset
each image (768x896) is split into 32x32 patches (672 patches per image) in a grid-fashion way
"""

import random
import torch
import glob
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import os
import utils_augmentation
from skimage import io
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt


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


def dataset_preview(dataset_dir):

    ds = BreastCancer(dataset_dir, keep_imgs=True, shuffle_bag=False)
    dl = data_utils.DataLoader(ds, shuffle=False)

    fig, axs = plt.subplots(nrows=10, ncols=6, figsize=(14, 20))

    for ax, batch in zip(axs.flatten(), dl):
        batch = [x.squeeze() for x in batch]
        _, label, img = batch

        ax.imshow(img)
        if label:
            plt.setp(ax.spines.values(), color="red", linewidth=3)
        else:
            plt.setp(ax.spines.values(), color="green", linewidth=3)
        ax.tick_params(axis="both",
                       which="both",
                       bottom=False,
                       top=False,
                       left=False,
                       right=False,
                       labelbottom=False,
                       labeltop=False,
                       labelleft=False,
                       labelright=False)

    # turn off unused axes completely
    for ax in axs.flatten()[len(ds):]:
        ax.set_axis_off()

    fig.tight_layout()
    plt.savefig(f"BreastCancer_preview.png")


def bag_preview(dataset_dir, idx=0):

    ds = BreastCancer(dataset_dir, shuffle_bag=True)
    bag, label, _ = ds[idx]

    fig, axs = plt.subplots(nrows=24, ncols=28, figsize=(28, 24))

    for ax, patch in zip(axs.flatten(), bag):
        ax.imshow(patch.permute(1, 2, 0))

        ax.tick_params(axis="both",
                       which="both",
                       bottom=False,
                       top=False,
                       left=False,
                       right=False,
                       labelbottom=False,
                       labeltop=False,
                       labelleft=False,
                       labelright=False)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.tight_layout()
    plt.savefig(f"bag_{idx}_preview_shuffled.png")


if __name__ == "__main__":
    import os
    ds_dir = os.path.join(os.environ["HOME"], "Repos", "data", "mil_datasets", "BreastCancer", "")
    # dataset_preview(ds_dir)
    # bag_preview(ds_dir)
