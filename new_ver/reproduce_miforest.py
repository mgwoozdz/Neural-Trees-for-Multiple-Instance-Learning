"""
This experiment aims to reproduce first two rows of table 1 of MIForest paper.
"""

import datasets
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from torch.utils.data import Subset, DataLoader


def bags_to_instances(bags, labels):
    """
    Helper function to unpack instances from multiple bags. Every instance is assigned the bag label it belongs to.
    """
    instances = []
    instances_y = []

    for bag, label in zip(bags, labels):
        for instance in bag:
            instances.append(instance)
            instances_y.append(label)

    return instances, instances_y


def reproduce_random_forest(ds_name):
    # as in the paper, train random forest dropping MIL constraint (i.e. on instances, dropping bags abstraction)
    # but still we should score the model under MIL constraint for fair comparison

    ds, _ = datasets.get_datasets(ds_name)
    kf = KFold(n_splits=5, shuffle=True)
    model = RandomForestClassifier(n_estimators=50, max_depth=20)

    i_acc, b_acc = 0, 0
    for train_idxs, test_idxs in kf.split(range(len(ds))):
        # since kf.split takes X of shape (n_samples, n_features) and ds.bags contains list of bags,
        # i.e. list of list of instances, possibly different length, we pass

        bags_train = np.array(ds.bags, dtype=object)[train_idxs]
        labels_train = np.array(ds.labels, dtype=object)[train_idxs]
        x_train, y_train = bags_to_instances(bags_train, labels_train)

        model.fit(x_train, y_train)

        # scoring
        bags_test = np.array(ds.bags, dtype=object)[test_idxs]
        labels_test = np.array(ds.labels, dtype=object)[test_idxs]

        # based on instances
        x_test, y_test = bags_to_instances(bags_test, labels_test)
        i_acc += model.score(x_test, y_test)

        # based on bags
        tmp_b_acc = 0
        for bag, label in zip(bags_test, labels_test):
            pred = model.predict(bag)
            # np.any(pred) returns True if model classifies as positive (at least one 1) and False otherwise
            tmp_b_acc += 1 if (np.any(pred) == bool(label)) else 0
        tmp_b_acc /= len(labels_test)
        b_acc += tmp_b_acc

    i_acc /= kf.get_n_splits()
    b_acc /= kf.get_n_splits()
    print(f"instance-based: {i_acc:.3f} | bag-based: {b_acc:.3f} \t ({ds_name})")


def reproduce_miforest(ds_name):
    pass


def run_experiment():
    for ds_name in ["elephant", "fox", "tiger", "musk1", "musk2"]:
        reproduce_random_forest(ds_name)
        reproduce_miforest(ds_name)


if __name__ == "__main__":
    run_experiment()