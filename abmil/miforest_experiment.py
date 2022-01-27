import numpy as np

from models.random_forest.tree import MIDecisionTree
from datasets.utils import ColonCancerDataset

dataset = ColonCancerDataset(root_dir="datasets/colon_cancer")


bag_indices, reprs, labels = dataset.read_reprs_from_folder(
    folder="datasets/colon_cancer_reprs"
)
print(bag_indices[:5])
print(reprs.shape, labels.shape)
print(np.bincount(labels))
print(len(bag_indices))

tree = MIDecisionTree()
tree.fit(reprs, labels, bag_indices[:80])

print("MI Random tree oob score: ", tree.oob_score)
