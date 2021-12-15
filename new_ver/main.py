import sklearn.model_selection
import datasets
import models
import numpy as np

plain_dataset, _ = datasets.get_datasets("musk1")

train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(plain_dataset.bags,
                                                                                  plain_dataset.labels,
                                                                                  test_size=0.1,
                                                                                  shuffle=True,
                                                                                  random_state=420)


flat_train = []
flat_train_labels = []
for bag, label in zip(train, train_labels):
    for instance in bag:
        flat_train.append(instance)
        flat_train_labels.append(label)


model = models.get_model("mif",
                         forest_size=100,
                         bags=train,
                         bag_labels=train_labels,
                         stop_temp=0.005)

model.train()

print(model.test(test, test_labels))
