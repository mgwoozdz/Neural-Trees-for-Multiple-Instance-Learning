"""
This experiment aims to reproduce rows of tables 2 and 3 corresponding to ABMIL (not gated).
"""
import os.path

import numpy as np
import torch
import datasets
import models
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader

# utils_augmentations clipping error
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


def fit_abmil(ds_name, train_ds, test_ds, gated=False, lr=1e-4, weight_decay=5e-4, epochs=50, seed=420420):

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    for ith_split, (train_idxs, test_idxs) in enumerate(skf.split(train_ds.bags, train_ds.labels), start=1):

        # prepare loaders
        train_loader = DataLoader(Subset(train_ds, train_idxs), batch_size=1, shuffle=True)
        test_loader = DataLoader(Subset(test_ds, test_idxs), batch_size=1, shuffle=False)

        # prepare model
        model = models.get_model("abmil", ds_name=ds_name, gated=gated)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

        # train
        model.fit(train_loader, test_loader, optimizer, epochs)

        # save best model
        dir_path = os.path.join("models", "saved_models", "")
        os.makedirs(dir_path, exist_ok=True)
        filename = f"{ds_name}{'_gated_' if gated else '_'}{ith_split}.pt"
        torch.save(model.bmk.state_dict, dir_path + filename)


def score_abmil(ds_name, train_ds, test_ds, gated=False, seed=420420):

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    results = {"accuracy": [], "precision": [], "recall": [], "f-score": [], "auc": []}
    for ith_split, (train_idxs, test_idxs) in enumerate(skf.split(train_ds.bags, train_ds.labels), start=1):

        test_loader = DataLoader(Subset(test_ds, test_idxs), batch_size=1, shuffle=False)
        model = models.get_model("abmil", ds_name=ds_name, gated=gated)
        path = os.path.join("models", "saved_models", f"{ds_name}_{ith_split}.pt")
        model.load_state_dict(torch.load(path))
        model.score(test_loader, dict_handle=results)

    return results


def reproduce_abmil():
    ds_names = ["breast_cancer", "colon_cancer"]

    # for ds_name in ds_names:
    #     plain_dataset, augmented_dataset = datasets.get_datasets(ds_name)
    #     fit_abmil(ds_name, train_ds=augmented_dataset, test_ds=plain_dataset)

    results = {}
    for ds_name in ds_names:
        plain_dataset, augmented_dataset = datasets.get_datasets(ds_name)
        results[ds_name] = score_abmil(ds_name, train_ds=augmented_dataset, test_ds=plain_dataset)

    print("dataset\t\t\t __accuracy__\t\t__precision__\t\t___recall___\t\t___F-score___\t\t_____AUC_____")
    for ds_name in ds_names:
        print(f"{ds_name}\t\t",
              f"{np.mean(results[ds_name]['accuracy']):.3f} ({np.std(results[ds_name]['accuracy']):.3f})\t\t"
              f"{np.mean(results[ds_name]['precision']):.3f} ({np.std(results[ds_name]['precision']):.3f})\t\t"
              f"{np.mean(results[ds_name]['recall']):.3f} ({np.std(results[ds_name]['recall']):.3f})\t\t"
              f"{np.mean(results[ds_name]['f-score']):.3f} ({np.std(results[ds_name]['f-score']):.3f})\t\t"
              f"{np.mean(results[ds_name]['auc']):.3f} ({np.std(results[ds_name]['auc']):.3f})\t\t")


if __name__ == "__main__":
    reproduce_abmil()
