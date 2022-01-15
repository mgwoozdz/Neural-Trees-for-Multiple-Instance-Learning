import torch
from tqdm import tqdm

import datasets
import models
import loops
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


# negative log bernoulli
def objective_fn(y_prob, target):
    target = torch.tensor(target, dtype=torch.float)
    return -1. * (target * torch.log(y_prob) + (1. - target) * torch.log(1. - y_prob))


# accuracy
def error_fn(y_hat, target):
    return y_hat.eq(target).cpu().float().mean().item()


def reproduce_abmil(ds_name, epochs=2):
    plain_dataset, augmented_dataset = datasets.get_datasets(ds_name)
    skf = StratifiedKFold(n_splits=10)

    avg_acc = 0
    for nth_split, (train_idxs, val_idxs) in enumerate(skf.split(range(len(plain_dataset)), plain_dataset.labels)):
        print(f"kfold {nth_split + 1}/{skf.n_splits}")

        train_loader = DataLoader(Subset(augmented_dataset, train_idxs), batch_size=1, shuffle=True)
        valid_loader = DataLoader(Subset(plain_dataset, val_idxs), batch_size=1, shuffle=False)

        model = models.get_model("abmil")
        optimizer = torch.optim.Adam(model.parameters(), lr=10e-4, betas=(0.9, 0.999), weight_decay=10e-5)

        last_epoch_acc = 0
        for e in tqdm(range(epochs)):
            tqdm.write(f"epoch {e}", end="\t | \t")
            train_loss, _ = loops.train(train_loader, model, optimizer, objective_fn, error_fn)
            tqdm.write(f"train loss:\t{train_loss}", end="\t | \t")
            _, valid_error = loops.test(valid_loader, model, objective_fn, error_fn)
            tqdm.write(f"valid accuracy:\t{valid_error}")
            last_epoch_acc = valid_error
        avg_acc += last_epoch_acc

    avg_acc /= skf.get_n_splits()
    print(f"average accuracy: {avg_acc:.3f} \t ({ds_name})")


def run_experiment():
    for ds_name in ["breast_cancer"]:   # , "colon_cancer"
        reproduce_abmil(ds_name)


if __name__ == "__main__":
    run_experiment()
