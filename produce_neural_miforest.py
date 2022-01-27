import sklearn
import numpy as np
from torch.utils.data import DataLoader, Subset

import datasets
from models.DNDF import DNDF


def produce_neural_miforest(ds_name):
    ds, _ = datasets.get_datasets(ds_name)
    print(len(ds))
    train_idx, test_idx, _, _ = sklearn.model_selection.train_test_split(np.arange(len(ds)),
                                                                         ds.bag_labels,
                                                                         test_size=0.1,
                                                                         shuffle=True, )

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=1, shuffle=True)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=1, shuffle=True)

    model = DNDF(forest_size=10, dataloader=train_loader, test_loader=test_loader,
                 stop_temp=0.005, n_in_feature=500)

    model.train()



if __name__ == "__main__":
    produce_neural_miforest("breast_cancer_reprs")
