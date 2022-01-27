import numpy as np


def kfold_indices_warwick(N, k, seed=777):
    r = np.random.RandomState(seed)
    all_indices = np.arange(N, dtype=int)
    r.shuffle(all_indices)
    idx = [int(i) for i in np.floor(np.linspace(0, N - 1, k + 1))]
    train_folds = []
    valid_folds = []
    for fold in range(k):
        valid_indices = all_indices[idx[fold] : idx[fold + 1]]
        valid_folds.append(valid_indices)
        train_fold = np.setdiff1d(all_indices, valid_indices)
        r.shuffle(train_fold)
        train_folds.append(train_fold)
    return train_folds, valid_folds


def train_val_test_split_warwick(N, k, seed=777):
    global_train_folds, test_folds = kfold_indices_warwick(N, k, seed=seed)
    local_train_folds, local_val_folds = kfold_indices_warwick(
        len(global_train_folds[0]), k, seed=seed
    )
    local_train_folds = [
        global_train_folds[0][indices] for indices in local_train_folds
    ]
    local_val_folds = [global_train_folds[0][indices] for indices in local_val_folds]
    return local_train_folds, local_val_folds, test_folds
