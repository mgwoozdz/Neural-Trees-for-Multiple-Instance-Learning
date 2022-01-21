import sklearn
import numpy as np
from torch.utils.data import DataLoader, Subset

import datasets
from models.DNDF import DNDF


def produce_neural_miforest(ds_name):
    ds, _ = datasets.get_datasets(ds_name)
    train_idx, test_idx, _, _ = sklearn.model_selection.train_test_split(np.arange(len(ds.bags)),
                                                                         ds.labels,
                                                                         test_size=0.1,
                                                                         shuffle=True, )

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=1, shuffle=True)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=1, shuffle=True)

    model = DNDF(forest_size=50, dataloader=train_loader,
                 stop_temp=0.005)

    model.train()
    print(ds_name + ':', model.test(test_loader))


def run_experiment():
    for ds_name in ["elephant"]:
        produce_neural_miforest(ds_name)


if __name__ == "__main__":
    produce_neural_miforest("musk1")
