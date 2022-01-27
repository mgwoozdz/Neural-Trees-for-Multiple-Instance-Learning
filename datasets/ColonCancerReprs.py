import numpy as np
import torch
import torch.utils.data as data_utils
from datasets.utils.colon_cancer import ColonCancerDataset


def make_bag_labels(
    X_bag_indices_ranges: np.array, y_instance_labels: np.array
) -> np.array:
    return np.asarray(
        [np.max(y_instance_labels[i[0] : i[1]]) for i in X_bag_indices_ranges]
    )


class ColonCancerReprs(data_utils.Dataset):

    def __init__(self):

        dataset = ColonCancerDataset(root_dir="...")

        X_bag_indices, X_reprs, y_reprs_labels = dataset.read_reprs_from_folder(
            folder="datasets/data/colon_cancer_reprs"
        )

        self.bags_indices = X_bag_indices
        self.bag_labels = make_bag_labels(X_bag_indices, y_reprs_labels)
        self.instances_reprs = X_reprs
        self.instances_labels = y_reprs_labels

    def __len__(self):
        return len(self.bags_indices)

    def __getitem__(self, idx):
        start, end = self.bags_indices[idx]
        return torch.tensor(self.instances_reprs[start: end]), self.bag_labels[idx]