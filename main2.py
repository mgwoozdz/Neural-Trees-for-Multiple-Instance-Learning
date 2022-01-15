import sklearn.model_selection
import datasets
import models
import numpy as np
from torch.utils.data import Subset, DataLoader

plain_dataset, _ = datasets.get_datasets("musk1")

train_idx, test_idx, _, _ = sklearn.model_selection.train_test_split(np.arange(len(plain_dataset.bags)),
                                                                     plain_dataset.labels,
                                                                     test_size=0.1,
                                                                     shuffle=True,
                                                                     random_state=420)

train_loader = DataLoader(Subset(plain_dataset, train_idx), batch_size=1, shuffle=True)
test_loader = DataLoader(Subset(plain_dataset, test_idx), batch_size=1, shuffle=True)

model = models.get_model("mif",
                         forest_size=50,
                         dataloader=train_loader,
                         stop_temp=0.005)

model.train()

print(model.test(test_loader))
