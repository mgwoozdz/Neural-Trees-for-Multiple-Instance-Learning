from torch import optim
import torch
from tqdm import tqdm

import data
import models
import loops
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Subset, DataLoader


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# ds_name = "colon"
# ds_dir = "Data/ColonCancer/"

ds_name = "breast"
ds_dir = "Data/BreastCancer/"

model_name = "attention"
lr = 0.0001
wd = 0.00001
epochs = 5


# negative log bernoulli
def objective_fn(y_prob, target):
    target = torch.tensor(target, dtype=torch.float)
    return -1. * (target * torch.log(y_prob) + (1. - target) * torch.log(1. - y_prob))


# accuracy
def error_fn(y_hat, target):
    return y_hat.eq(target).cpu().float().mean().item()


# instantiate both plain and augmented datasets
plain_dataset = data.DatasetHolder(ds_name, ds_dir, augment=False)
augmented_dataset = data.DatasetHolder(ds_name, ds_dir, augment=True)

# split data into train+val and test sets
train_val_idxs, test_idxs, train_val_labels, test_labels = train_test_split(list(range(len(plain_dataset))),
                                                                            plain_dataset.labels,
                                                                            test_size=0.1,
                                                                            stratify=plain_dataset.labels)

# cross-validate over of train+val split
skf = StratifiedKFold(n_splits=10)

for nth_split, (train_idxs, val_idxs) in enumerate(skf.split(train_val_idxs, train_val_labels)):
    print(f"kfold {nth_split + 1}/{skf.n_splits}")

    train_loader = DataLoader(Subset(augmented_dataset, train_idxs), batch_size=1, shuffle=True)
    valid_loader = DataLoader(Subset(plain_dataset, val_idxs), batch_size=1, shuffle=False)

    model = models.get_model(model_name)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)

    for e in tqdm(range(epochs)):
        tqdm.write(f"epoch {e}", end="\t | \t")
        train_loss, _ = loops.train(train_loader, model, optimizer, objective_fn, error_fn)
        tqdm.write(f"train loss:\t{train_loss}", end="\t | \t")
        _, valid_error = loops.test(valid_loader, model, objective_fn, error_fn)
        tqdm.write(f"valid accuracy:\t{valid_error}")

# test on test split

# test_loader = DataLoader(Subset(plain_dataset, test_idxs), shuffle=False)
# loops.test()
