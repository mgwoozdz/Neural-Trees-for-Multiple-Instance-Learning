"""Pytorch Dataset object that loads 27x27 patches that contain single cells."""

import os
import random
from typing import Sequence, Tuple, List

import numpy as np
import scipy.io
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from PIL import Image
from numpy import ndarray
from skimage import color, io
import re
from collections import defaultdict
from pathlib import Path
from typing import Union

from torch import nn
from tqdm import tqdm

from . import augmentations
from .datasets import COLON_CANCER_CLASSIFICATION_PATH


class ColonCancerBagsCross(data_utils.Dataset):
    def __init__(
        self,
        path,
        train_val_idxs=None,
        test_idxs=None,
        train=True,
        shuffle_bag=False,
        data_augmentation=False,
        loc_info=False,
    ):
        self.path = path
        self.train_val_idxs = train_val_idxs
        self.test_idxs = test_idxs
        self.train = train
        self.shuffle_bag = shuffle_bag
        self.data_augmentation = data_augmentation
        self.location_info = loc_info

        self.data_augmentation_img_transform = transforms.Compose(
            [
                augmentations.RandomHEStain(),
                augmentations.HistoNormalize(),
                augmentations.RandomRotate(),
                augmentations.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.normalize_to_tensor_transform = transforms.Compose(
            [
                augmentations.HistoNormalize(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dir_list_train, self.dir_list_test = self.split_dir_list(
            self.path, self.train_val_idxs, self.test_idxs
        )
        if self.train:
            self.bag_list_train, self.labels_list_train = self.create_bags(
                self.dir_list_train
            )
        else:
            self.bag_list_test, self.labels_list_test = self.create_bags(
                self.dir_list_test
            )

    @staticmethod
    def split_dir_list(path, train_val_idxs, test_idxs):
        dirs = [x[0] for x in os.walk(path)]
        dirs.pop(0)
        dirs.sort()

        dir_list_train = [dirs[i] for i in train_val_idxs]
        dir_list_test = [dirs[i] for i in test_idxs]

        return dir_list_train, dir_list_test

    def create_bags(self, dir_list):
        bag_list = []
        labels_list = []
        for dir in dir_list:
            # Get image name
            img_name = dir.split("/")[-1]

            # bmp to pillow
            img_dir = dir + "/" + img_name + ".bmp"
            img = io.imread(img_dir)
            if img.shape[2] == 4:
                img = color.rgba2rgb(img)

            if self.location_info:
                xs = np.arange(0, 500)
                xs = np.asarray([xs for i in range(500)])
                ys = xs.transpose()
                img = np.dstack((img, xs, ys))

            # crop malignant cells
            dir_epithelial = dir + "/" + img_name + "_epithelial.mat"
            with open(dir_epithelial, "rb") as f:
                mat_epithelial = scipy.io.loadmat(f)

            cropped_cells_epithelial = []
            for (x, y) in mat_epithelial["detection"]:
                x = np.round(x)
                y = np.round(y)

                if self.data_augmentation:
                    x = x + np.round(np.random.normal(0, 3, 1))
                    y = y + np.round(np.random.normal(0, 3, 1))

                if x < 13:
                    x_start = 0
                    x_end = 27
                elif x > 500 - 14:
                    x_start = 500 - 27
                    x_end = 500
                else:
                    x_start = x - 13
                    x_end = x + 14

                if y < 13:
                    y_start = 0
                    y_end = 27
                elif y > 500 - 14:
                    y_start = 500 - 27
                    y_end = 500
                else:
                    y_start = y - 13
                    y_end = y + 14

                cropped_cells_epithelial.append(
                    img[int(y_start) : int(y_end), int(x_start) : int(x_end)]
                )

            # crop all other cells
            dir_inflammatory = dir + "/" + img_name + "_inflammatory.mat"
            dir_fibroblast = dir + "/" + img_name + "_fibroblast.mat"
            dir_others = dir + "/" + img_name + "_others.mat"

            with open(dir_inflammatory, "rb") as f:
                mat_inflammatory = scipy.io.loadmat(f)
            with open(dir_fibroblast, "rb") as f:
                mat_fibroblast = scipy.io.loadmat(f)
            with open(dir_others, "rb") as f:
                mat_others = scipy.io.loadmat(f)

            all_coordinates = np.concatenate(
                (
                    mat_inflammatory["detection"],
                    mat_fibroblast["detection"],
                    mat_others["detection"],
                ),
                axis=0,
            )

            cropped_cells_others = []
            for (x, y) in all_coordinates:
                x = np.round(x)
                y = np.round(y)

                if self.data_augmentation:
                    x = x + np.round(np.random.normal(0, 3, 1))
                    y = y + np.round(np.random.normal(0, 3, 1))

                if x < 13:
                    x_start = 0
                    x_end = 27
                elif x > 500 - 14:
                    x_start = 500 - 27
                    x_end = 500
                else:
                    x_start = x - 13
                    x_end = x + 14

                if y < 13:
                    y_start = 0
                    y_end = 27
                elif y > 500 - 14:
                    y_start = 500 - 27
                    y_end = 500
                else:
                    y_start = y - 13
                    y_end = y + 14

                cropped_cells_others.append(
                    img[int(y_start) : int(y_end), int(x_start) : int(x_end)]
                )

            # generate bag
            bag = cropped_cells_epithelial + cropped_cells_others

            # store single cell labels
            labels = np.concatenate(
                (
                    np.ones(len(cropped_cells_epithelial)),
                    np.zeros(len(cropped_cells_others)),
                ),
                axis=0,
            )

            # shuffle
            if self.shuffle_bag:
                zip_bag_labels = list(zip(bag, labels))
                random.shuffle(zip_bag_labels)
                bag, labels = zip(*zip_bag_labels)

            # append every bag two times if training
            if self.train:
                for _ in [0, 1]:
                    bag_list.append(bag)
                    labels_list.append(labels)
            else:
                bag_list.append(bag)
                labels_list.append(labels)

            # bag_list.append(bag)
            # labels_list.append(labels)

        return bag_list, labels_list

    def transform_and_data_augmentation(self, bag):
        if self.data_augmentation:
            img_transform = self.data_augmentation_img_transform
        else:
            img_transform = self.normalize_to_tensor_transform

        bag_tensors = []
        for img in bag:
            if self.location_info:
                bag_tensors.append(
                    torch.cat(
                        (
                            img_transform(img[:, :, :3]),
                            torch.from_numpy(
                                img[:, :, 3:].astype(float).transpose((2, 0, 1))
                            ).float(),
                        )
                    )
                )
            else:
                bag_tensors.append(img_transform(img))
        return torch.stack(bag_tensors)

    def __len__(self):
        if self.train:
            return len(self.labels_list_train)
        else:
            return len(self.labels_list_test)

    def __getitem__(self, index):
        if self.train:
            bag = self.bag_list_train[index]
            label = [max(self.labels_list_train[index]), self.labels_list_train[index]]
        else:
            bag = self.bag_list_test[index]
            label = [max(self.labels_list_test[index]), self.labels_list_test[index]]

        return self.transform_and_data_augmentation(bag), label


def kth_train_val_test_data_loaders(
    train_folds: Sequence,
    val_folds: Sequence,
    test_folds: Sequence,
    k: int,
    batch_size: int = 1,
):
    return {
        "train": data_utils.DataLoader(
            ColonCancerBagsCross(
                COLON_CANCER_CLASSIFICATION_PATH,
                train_val_idxs=train_folds[k],
                test_idxs=test_folds[k],
                train=True,
                shuffle_bag=True,
                data_augmentation=True,
            ),
            batch_size=batch_size,
            shuffle=True,
        ),
        "val": data_utils.DataLoader(
            ColonCancerBagsCross(
                COLON_CANCER_CLASSIFICATION_PATH,
                train_val_idxs=val_folds[k],
                test_idxs=test_folds[k],
                train=True,
                shuffle_bag=True,
                data_augmentation=True,
            ),
            batch_size=batch_size,
            shuffle=True,
        ),
        "test": data_utils.DataLoader(
            ColonCancerBagsCross(
                COLON_CANCER_CLASSIFICATION_PATH,
                train_val_idxs=val_folds[k],
                test_idxs=test_folds[k],
                train=False,
                shuffle_bag=False,
                data_augmentation=False,
            ),
            batch_size=batch_size,
            shuffle=False,
        ),
    }


def load_image(image_path: Union[str, Path]) -> np.array:
    with Image.open(image_path) as img:
        img.load()
        return np.asarray(img, dtype="float32")


def read_mat_file(mat_file_path: Union[str, Path]) -> dict:
    return scipy.io.loadmat(mat_file_path)


class ColonCancerDataset(data_utils.Dataset):
    index_regex = re.compile(r".*/?img(\d+)")
    npy_repr_regex = re.compile(r"^.*/(\d+)\.npy$")
    image_repr_path_index_regex = re.compile(r"^.*/?(\d+)$")

    REPR_DIM = 500

    transform = transforms.Compose(
        [
            augmentations.HistoNormalize(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    nuclei_classes = [
        "epithelial",
        "fibroblast",
        "inflammatory",
        "others",
    ]

    @property
    def classification_path(self) -> Path:
        return Path(self.root_dir) / "CRCHistoPhenotypes_2016_04_28" / "Classification"

    def __read_data_from_folder(self):
        for path in self.pathlist:
            index = self.index_regex.match(str(path))
            if index:
                index = int(index.groups()[0])
                img_path = path / f"img{index}.bmp"
                img = io.imread(str(img_path))
                if img.shape[2] == 4:
                    img = color.rgba2rgb(img)
                self.images[index - 1] = img
                for nuclei_class in self.nuclei_classes:
                    mat = read_mat_file(path / f"img{index}_{nuclei_class}.mat")
                    self.index2nuclei[index - 1][nuclei_class] = mat["detection"]
                self.labels[index - 1] = np.array(
                    bool(self.index2nuclei[index - 1]["epithelial"].size)
                )

    def save_reprs_to_folder(
        self, model: nn.Module, folder: Union[str, Path], device="cpu"
    ):
        model = model.to(device)
        save_path = Path(folder) if isinstance(folder, str) else folder
        for i, image in tqdm(enumerate(self.images)):
            img_path = save_path / str(i + 1)
            img_path.mkdir(parents=True, exist_ok=True)
            nuclei = self.index2nuclei[i]
            for nuclei_class, nuclei_list in nuclei.items():
                nuclei_path = img_path / nuclei_class
                nuclei_path.mkdir(parents=True, exist_ok=True)
                for j, (x, y) in enumerate(nuclei_list):
                    x = np.round(x)
                    y = np.round(y)

                    if x < 13:
                        x_start = 0
                        x_end = 27
                    elif x > 500 - 14:
                        x_start = 500 - 27
                        x_end = 500
                    else:
                        x_start = x - 13
                        x_end = x + 14

                    if y < 13:
                        y_start = 0
                        y_end = 27
                    elif y > 500 - 14:
                        y_start = 500 - 27
                        y_end = 500
                    else:
                        y_start = y - 13
                        y_end = y + 14

                    patch = self.transform(
                        image[int(y_start) : int(y_end), int(x_start) : int(x_end)]
                    )
                    patch = torch.unsqueeze(torch.unsqueeze(patch, 0), 0)
                    patch = patch.to(device)
                    patch_repr: torch.Tensor = model(patch)
                    patch_repr = patch_repr.detach()
                    np.save(nuclei_path / str(j), np.asarray(np.asarray(patch_repr)))

    def read_reprs_from_folder(
        self, folder: Union[str, Path]
    ) -> Tuple[ndarray, ndarray, ndarray]:
        reprs_path = Path(folder) if isinstance(folder, str) else folder
        path_list = list(Path(reprs_path).glob("*"))
        bag_indices: List[List[int, int]] = []
        num_of_instances = 0
        for path in path_list:
            for nuclei_class in self.nuclei_classes:
                num_of_instances += len(list((path / nuclei_class).glob("*.npy")))

        reprs = np.zeros((num_of_instances, self.REPR_DIM))
        labels = np.zeros(num_of_instances)
        i = 0
        for path in tqdm(path_list):
            bag_start = i
            for nuclei_class in self.nuclei_classes:
                nuclei_path_list = list((path / nuclei_class).glob("*.npy"))
                for nuclei_path in nuclei_path_list:
                    reprs[i] = np.load(str(nuclei_path)).reshape(1, -1)
                    labels[i] = int(nuclei_class == "epithelial")
                    i += 1
            bag_indices.append([bag_start, i])
        return np.asarray(bag_indices), reprs, np.asarray(labels, dtype="int8")

    def __init__(
        self, root_dir: str = "colon_cancer", transform=None, target_transform=None
    ):
        self.root_dir = root_dir
        self.transform = transform or self.__class__.transform
        self.target_transform = target_transform
        self.pathlist = list(Path(self.classification_path).glob("img*/"))
        self.images = np.zeros((len(self.pathlist), 500, 500, 3))
        self.labels = np.zeros((len(self.pathlist), 1))
        self.index2nuclei = defaultdict(lambda: defaultdict(lambda: np.array([])))
        self.__read_data_from_folder()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.transform(self.images[index]), self.labels[index]
