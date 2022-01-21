import time

import numpy as np
import sklearn
from sklearn import ensemble
from scipy.optimize import minimize
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F

# %%
from models.neural_forest.Forest import Forest
from models.neural_forest.NeuralForest import NeuralForest

n_tree = 5
tree_depth = 3
n_class = 10
tree_feature_rate = 0.5


# %%
class Config:
    def __init__(self):
        self.jointly_training = False
        self.epochs = 10
        self.batch_size = 64
        self.report_every = 10
        self.n_class = 10
        self.cuda = True
        self.seed = 1


class DNDF:
    def __init__(self, forest_size, dataloader, start_temp=None, stop_temp=None):
        self.forests_size = forest_size
        self.dataloader = dataloader
        self.start_temp = start_temp
        self.stop_temp = stop_temp
        self.init_y = []

        forest = Forest(n_tree=forest_size, tree_depth=tree_depth, n_in_feature=784,
                        tree_feature_rate=tree_feature_rate, n_class=n_class,
                        jointly_training=True)

        self.neural_forest = NeuralForest(forest, Config())
        self.optim = torch.optim.Adam(self.neural_forest.get_params(), lr=0.01)
        # self.random_forest = ensemble.RandomForestClassifier(n_estimators=forest_size,
        #                                                      max_depth=20,
        #                                                      max_features=25)

    @staticmethod
    def prepare_data(loader):
        # we will process instances directly

        instances = []
        instances_y = []

        for bag, label, _ in loader:
            for instance in bag[0]:
                instances.append(np.array(instance).flatten())
                instances_y.append(label.item())
        return instances, instances_y

    @staticmethod
    def cooling_fn(epoch, const=0.5):
        return np.exp(-epoch * const)

    def calc_p_star(self, p_hat, preds, temp):
        # equation 8 (section 3.1)
        loss_term = p_hat * (4 * preds - 2) + 2 * preds - 1
        temp_term = p_hat * np.log(p_hat) + (1 - p_hat) * np.log(1 - p_hat)
        return np.sum(loss_term - temp * temp_term)

    def train(self, tol=1e-6):

        # step 1 - train random forest using the bag label as label of instances

        instances, instances_y = self.prepare_data(self.dataloader)
        # print(type(instances), type(instances_y))

        self.init_y = instances_y
        self.neural_forest.train_full(self.optim, instances, instances_y)

        # step 2 - retrain trees substituting labels

        epoch = 0
        temp = self.cooling_fn(epoch)
        while temp > self.stop_temp:

            temp = self.cooling_fn(epoch)

            preds = self.neural_forest.forest.forward(instances)[:, 0]

            start = time.time()
            probs = minimize(fun=self.calc_p_star,
                             x0=np.full(len(instances), 0.5),
                             args=(preds, temp),
                             bounds=np.tile([tol, 1 - tol], (len(instances), 1)),
                             method='SLSQP')
            end = time.time()

            print(f"epoch {epoch}: minimize took {end - start:.3f}s", end=", ")

            start = time.time()
            for tree in self.neural_forest.forest.trees:
                for i, (prob, y) in enumerate(zip(probs.x, instances_y)):
                    instances_y[i] = np.random.choice([0, 1], p=(prob, 1 - prob))

                highest_idx = np.argmax(probs)
                instances_y[highest_idx] = self.init_y[highest_idx]

                tree.fit(instances, instances_y)

            epoch += 1
            end = time.time()
            print(f"fitting trees took {end - start:.3f}s")

    def predict(self, examples):
        return self.neural_forest.forest.forward(examples)

    def test(self, loader):
        instances, instances_y = self.prepare_data(loader)

        pred = self.predict(instances)
        return sklearn.metrics.accuracy_score(instances_y, pred)
