"""
intended use:
- call get_datasets to obtain train and val_test dataset classes for given name
- prepare cross-validation splits of indices (you need len(dataset) to do so)
- call get_dataloaders to obtain train, val and test dataloader classes for given indices
"""
import sklearn.model_selection
from torch.utils.data import DataLoader, Subset

from .BreastCancer import BreastCancer
from .ColonCancer import ColonCancer
from .Classic import Classic
from .MNIST_bags import MNIST_Bags


def get_datasets(name):
    """
    for given dataset name returns its plain and augmented dataset class
    """
    if name == "breast_cancer":

        basic_ds = BreastCancer(augment=False)
        augmented_ds = BreastCancer(augment=True)

    elif name in ["elephant", "fox", "musk1", "musk2", "tiger"]:
        # in this case we do not use augmentations
        dataset = Classic(name)
        basic_ds, augmented_ds = dataset, dataset

    # elif name == "colon_cancer":

    # elif name == "mnist_bags":

    else:
        raise Exception(f"dataset {name} not defined")

    return basic_ds, augmented_ds
