"""
we converted matlab versions http://www.cs.columbia.edu/~andrews/mil/datasets.html to csv
"""


import numpy as np

from torch.utils.data import Dataset


class Classic(Dataset):

    def __init__(self,
                 name,
                 root_dir="datasets/data/classics/",
                 shuffle_bag=True):

        self.bags = []
        self.labels = []
        self.shuffle_bag = shuffle_bag
        self.create_bags(root_dir + name)

    def create_bags(self, data_dir):
        bag_ids = np.loadtxt(data_dir + "/bag_ids.csv", delimiter=',')
        features = np.loadtxt(data_dir + "/features.csv", delimiter=',')
        labels = np.loadtxt(data_dir + "/labels.csv", delimiter=',')

        for bag_i, label in enumerate(labels, start=1):
            self.bags.append(features[bag_ids == bag_i])
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        if self.shuffle_bag:
            np.random.shuffle(bag)

        label = self.labels[idx]
        return bag, label, False
