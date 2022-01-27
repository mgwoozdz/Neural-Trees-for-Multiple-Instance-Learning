import time

import numpy as np
from pygments import highlight
import sklearn
from sklearn import ensemble
from scipy.optimize import minimize
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm

# %%
from models.neural_forest.Forest import Forest
from models.neural_forest.NeuralForest import NeuralForest

tree_depth = 6
n_class = 2
tree_feature_rate = 0.5


class Config:
    def __init__(self):
        self.jointly_training = False
        self.epochs = 10
        self.batch_size = 64
        self.report_every = 10
        self.n_class = 2
        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print()

        # Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
        self.device = device
        self.seed = 1


class DNDF:
    def __init__(self, forest_size, dataloader, test_loader, start_temp=None, stop_temp=None,
                 n_in_feature=784):
        self.forests_size = forest_size
        self.dataloader = dataloader
        self.test_loader = test_loader
        self.start_temp = start_temp
        self.stop_temp = stop_temp
        self.init_y = []
        self.opt = Config()

        forest = Forest(n_tree=forest_size, tree_depth=tree_depth, n_in_feature=n_in_feature,
                        tree_feature_rate=tree_feature_rate, n_class=n_class,
                        jointly_training=True).to(self.opt.device)

        self.neural_forest = NeuralForest(forest, self.opt)
        self.optim = torch.optim.Adam(self.neural_forest.get_params(), lr=0.01)

    def prepare_data(self, loader):
        # we will process instances directly

        instances = []
        instances_y = []
        bags = []
        start = 0

        for bag, label, _ in loader:
            bags.append((start, bag.shape[1]))
            start += bag.shape[1]
            for instance in bag[0]:
                instances.append(np.array(instance).flatten())
                instances_y.append(label.item())

        instances, instances_y = torch.tensor(np.array(instances).astype(np.float32)), torch.tensor(
            np.array(instances_y).astype(np.int64))
        instances, instances_y = instances.to(self.opt.device), instances_y.to(self.opt.device)
        return bags, instances, instances_y

    @staticmethod
    def cooling_fn(temp, m=0.5):
        return np.exp(-temp * m)

    def train(self, tol=1e-6):
        # step 1 - train random forest using the bag label as label of instances
        bags, instances, instances_y = self.prepare_data(self.dataloader)

        test_bags, test_instances, test_instances_y = self.prepare_data(self.test_loader)

        self.init_y = instances_y
        self.neural_forest.train_full(self.optim, instances, instances_y)
        print("ACC:", self.test(instances, instances_y))

        # step 2 - retrain trees substituting labels
        epoch = 0
        temp = self.cooling_fn(epoch)
        while temp > self.stop_temp:
            print("EPOCH: ", epoch)

            temp = self.cooling_fn(epoch)
            epoch += 1

            preds = self.neural_forest.forest.forward(instances)[:, 0].cpu().detach().numpy()
            preds = np.clip((preds - 0.5) * 2.5 + 0.5, 0, 1)

            print(preds[:10])
            print("Temp:", temp)

            confidence = np.abs(0.5 - preds)

            start = time.time()
            losses = []
            with tqdm.trange(len(self.neural_forest.forest.trees)) as t:
                for idx in t:
                    for i, (prob, y) in enumerate(zip(preds, instances_y)):
                        instances_y[i] = np.random.choice([0, 1], p=(prob, 1 - prob))

                    highest_idx = np.argmax(confidence)
                    instances_y[highest_idx] = self.init_y[highest_idx]

                    for i in range(2):
                        loss = self.neural_forest.train_tree(idx, instances.to(self.opt.device),
                                                             instances_y.to(self.opt.device), temp,
                                                             self.optim)
                    losses.append(loss)
                    t.set_description(f"Loss {np.mean(losses):.3f}")

            end = time.time()
            print(
                f"fitting trees took {end - start:.3f}s, acc: {self.test(test_instances, test_instances_y):.3f}")

    def predict(self, examples):
        return self.neural_forest.forest.forward(examples)

    def test(self, instances, instances_y):
        instances_y = instances_y.cpu().detach().numpy()
        pred = self.predict(instances).cpu().detach().numpy()
        pred = np.argmax(pred, axis=1)
        return sklearn.metrics.accuracy_score(instances_y, pred)
