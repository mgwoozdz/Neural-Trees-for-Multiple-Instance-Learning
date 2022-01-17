"""
We aim to reproduce two bottom rows of tables 2 and 3.
"""
import os.path

import numpy as np
import torch
import datasets
import models
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from torch.utils.data import Subset, DataLoader


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


def fit_abmil(ds_name, gated, train_ds, test_ds, lr=1e-4, weight_decay=5e-4, epochs=50):

    # skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=420)
    skf = StratifiedKFold(n_splits=10)
    results = {"accuracy": [], "precision": [], "recall": [], "f-score": [], "auc": []}

    for ith_split, (train_idxs, test_idxs) in enumerate(skf.split(train_ds.bags, train_ds.labels), start=1):

        # prepare loaders
        train_loader = DataLoader(Subset(train_ds, train_idxs), batch_size=1, shuffle=True)
        test_loader = DataLoader(Subset(test_ds, test_idxs), batch_size=1, shuffle=False)

        # prepare model
        model = models.get_model("abmil", ds_name=ds_name, gated=gated)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

        # train and evaluate
        model.fit(train_loader, optimizer, epochs)
        model.score(test_loader, dict_handle=results)

        # save model
        dir_path = os.path.join("models", "saved_models", "")
        os.makedirs(dir_path, exist_ok=True)
        filename = f"{ds_name}{'_gated_' if gated else '_'}{ith_split}.pt"
        torch.save(model.state_dict(), dir_path + filename)

    return results


def reproduce_abmil():

    # TODO: add colon_cancer, (27, 27, 3)
    # for ds_name in ["breast_cancer", "colon_cancer"]:

    all_results = {}
    for ds_name in ["breast_cancer"]:
        plain_dataset, augmented_dataset = datasets.get_datasets(ds_name)

        all_results[ds_name] = {}
        handle = all_results[ds_name]
        for gated in [False, True]:
            handle[gated] = fit_abmil(ds_name, gated=gated, train_ds=augmented_dataset, test_ds=plain_dataset)

    for ds_name, v in all_results.items():
        print(f"\n{ds_name} results:".upper())
        print("gated\t\t __accuracy__\t\t__precision__\t\t___recall___\t\t___F-score___\t\t____AUC____")
        for gated, results in v.items():
            print(f"{'yes' if gated else 'no'}\t\t",
                  f"{np.mean(results['accuracy']):.3f} ({np.std(results['accuracy']):.3f})\t\t"
                  f"{np.mean(results['precision']):.3f} ({np.std(results['precision']):.3f})\t\t"
                  f"{np.mean(results['recall']):.3f} ({np.std(results['recall']):.3f})\t\t"
                  f"{np.mean(results['f-score']):.3f} ({np.std(results['f-score']):.3f})\t\t"
                  f"{np.mean(results['auc']):.3f} ({np.std(results['auc']):.3f})\t\t")


if __name__ == "__main__":
    reproduce_abmil()
