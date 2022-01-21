"""
This experiment aims to reproduce first two rows of table 1 of MIForest paper.
"""
import sklearn
from sklearn.metrics import accuracy_score

import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from models.MIForestMinimaliation import MIForest
import numpy as np

import warnings

warnings.simplefilter(action='ignore', category=RuntimeWarning)


def instance_score(model, instances, labels):
    return accuracy_score(labels, model.predict(instances))


def bag_score(model, bags, labels):
    return len([1 for b, l in zip(bags, labels) if (np.any(model.predict(b)) == bool(l))]) / len(labels)


def reproduce_miforest():
    # TODO: ds_names = ["elephant", "fox", "tiger", "musk1", "musk2"]
    ds_names = ["elephant", "fox", "tiger", "musk1"]
    # ds_names = ["musk1"]

    # prepare results holder
    results = {}
    for ds_name in ds_names:
        results[ds_name] = {"random_forest": {"instance_score": [],
                                              "bag_score": []},
                            "miforest": {"instance_score": [],
                                         "bag_score": []}}

    for ds_name in ds_names:
        for seed in range(321897301, 321897301 + 5):
            # prepare splits
            ds, _ = datasets.get_datasets(ds_name)
            train_idxs, test_idxs, _, _ = train_test_split(np.arange(len(ds.bags)),
                                                           ds.labels,
                                                           test_size=0.2,
                                                           shuffle=True)

            bags_train = np.array(ds.bags, dtype=object)[train_idxs]
            labels_train = np.array(ds.labels, dtype=object)[train_idxs]

            bags_test = np.array(ds.bags, dtype=object)[test_idxs]
            labels_test = np.array(ds.labels, dtype=object)[test_idxs]

            x_train, y_train = MIForest.bags_to_instances(bags_train, labels_train)
            x_test, y_test = MIForest.bags_to_instances(bags_test, labels_test)

            # prepare and train models
            random_forest = RandomForestClassifier()
            random_forest.fit(x_train, y_train)

            miforest = MIForest()
            miforest.fit(x_train, y_train)

            # log two-way evaluation results
            results[ds_name]["random_forest"]["instance_score"].append(instance_score(random_forest, x_test, y_test))
            results[ds_name]["random_forest"]["bag_score"].append(bag_score(random_forest, bags_test, labels_test))

            results[ds_name]["miforest"]["instance_score"].append(instance_score(miforest, x_test, y_test))
            results[ds_name]["miforest"]["bag_score"].append(bag_score(miforest, bags_test, labels_test))

    for score in ["instance_score", "bag_score"]:
        print(score.upper())
        print(f"method\t\t\t" + "\t\t".join(map(lambda s: s[:7], ds_names)))
        for method in ["random_forest", "miforest"]:
            print(f"{method}\t\t" + "\t\t".join(
                [f"{np.mean(results[ds_name][method][score]):.3f}" for ds_name in ds_names]))


if __name__ == "__main__":
    reproduce_miforest()

    # ds_names = ["elephant", "fox", "tiger", "musk1", "musk2"]
    # results = {}
    # for ds_name in ds_names:
    #     results[ds_name] = {"random_forest": {"instance_score": np.random.randn(5),
    #                                           "bag_score": np.random.randn(5)},
    #                         "miforest": {"instance_score": np.random.randn(5),
    #                                      "bag_score": np.random.randn(5)}}
    #
    # for score in ["instance_score", "bag_score"]:
    #     print(score.upper())
    #     print(f"method\t\t\t" + "\t\t".join(map(lambda s: s[:7], ds_names)))
    #     for method in ["random_forest", "miforest"]:
    #         print(f"{method}\t\t" + "\t\t".join([f"{np.mean(results[ds_name][method][score]):.3f}" for ds_name in ds_names]))
