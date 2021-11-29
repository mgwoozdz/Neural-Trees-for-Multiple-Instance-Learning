import random
import torch
import glob
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import utils_augemntation
from skimage import io, color
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt


# since both ColonCancer and BreastCancer datasets are processed almost the same way,
# we implement single class to handle them both using some helper definitions:

IMG_FILE_EXTENSION = {
    "breast": "*tif",
    "colon": "*bmp"
}

PATCH_SIZE = {
    "breast": (32, 32),
    "colon": (25, 25)
}

CLASS_MARKER = {
    "breast": "malignant",
    "colon": "epithelial"
}

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


class DatasetHolder(data_utils.Dataset):
    """Pytorch Dataset object that loads patches that contain single cells."""

    def __init__(self,
                 dataset_name,
                 data_dir,
                 augment=False,
                 keep_imgs=False,
                 shuffle_bag=True):

        assert dataset_name in ["breast", "colon"], f"{dataset_name} not supported"

        self.bags = []
        self.labels = []
        self.imgs = []

        self.keep_imgs = keep_imgs
        self.shuffle_bag = shuffle_bag
        self.transform = AUGMENTED_TRANSFORMS if augment else BASIC_TRANSFORMS
        self.create_bags(data_dir,
                         IMG_FILE_EXTENSION[dataset_name],
                         PATCH_SIZE[dataset_name],
                         CLASS_MARKER[dataset_name])

    def create_bags(self, data_dir, extension, patch_size, class_marker):

        # load all data paths in sorted order
        paths = glob.glob(data_dir + extension)
        paths.sort()

        for path in paths:
            img = io.imread(path)

            if img.shape[2] == 4:
                img = color.rgba2rgb(img)

            if self.keep_imgs:
                self.imgs.append(img)

            bag = view_as_blocks(img, block_shape=(*patch_size, 3)).reshape(-1, *patch_size, 3)
            self.bags.append(bag)

            label = 1 if class_marker in path else 0
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


if __name__ == "__main__":

    ds_names = ["colon", "breast"]
    ds_dirs = ["Data/ColonCancer/", "Data/BreastCancer/"]

    for ds_name, ds_dir in zip(ds_names, ds_dirs):
        ds = DatasetHolder(ds_name, ds_dir, keep_imgs=True, shuffle_bag=False)

        fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(20, 20))

        for ax, (bag, label, img) in zip(axs.flatten(), iter(ds)):
            ax.imshow(img)
            if label:
                plt.setp(ax.spines.values(), color="red")
            else:
                plt.setp(ax.spines.values(), color="green")
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

        for ax in axs.flatten()[len(ds):]:
            ax.set_axis_off()

        fig.tight_layout()

        plt.savefig(f"{ds_name}_preview.png")
        plt.clf()
