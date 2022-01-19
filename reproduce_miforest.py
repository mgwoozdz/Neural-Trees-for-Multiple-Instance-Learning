"""
This experiment aims to reproduce first two rows of table 1 of MIForest paper.
"""
import sklearn
from torch.utils.data import DataLoader, Subset
import datasets
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from models.MIForestMinimaliation import MIForest
import numpy as np

def bags_to_instances(bags, labels):
    instances = []
    instances_y = []

    for bag, label in zip(bags, labels):
        for instance in bag:
            instances.append(instance)
            instances_y.append(label)

    return instances, instances_y


def reproduce_random_forest(ds_name):
    # as in the paper, train random forest dropping MIL constraint (i.e. on instances, dropping bags abstraction)

    ds, _ = datasets.get_datasets(ds_name)
    kf = KFold(n_splits=5, shuffle=True)
    model = RandomForestClassifier(n_estimators=50, max_depth=20)

    i_acc, b_acc = 0, 0
    for train_idxs, test_idxs in kf.split(range(len(ds))):

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
    ds, _ = datasets.get_datasets(ds_name)
    train_idx, test_idx, _, _ = sklearn.model_selection.train_test_split(np.arange(len(ds.bags)),
                                                                         ds.labels,
                                                                         test_size=0.1,
                                                                         shuffle=True,)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=1, shuffle=True)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=1, shuffle=True)
    model = MIForest(forest_size=50, dataloader=train_loader,
                         stop_temp=0.005)
    model.train()
    print(ds_name + ':', model.test(test_loader))


def run_experiment():
    for ds_name in ["elephant"]:
        #reproduce_random_forest(ds_name)
        reproduce_miforest(ds_name)


if __name__ == "__main__":
    reproduce_miforest("musk1")
