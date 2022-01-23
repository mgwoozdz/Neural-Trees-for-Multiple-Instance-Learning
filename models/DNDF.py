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

n_tree = 5
tree_depth = 4
n_class = 2
tree_feature_rate = 0.5


# %%
class Config:
    def __init__(self):
        self.jointly_training = False
        self.epochs = 10
        self.batch_size = 64
        self.report_every = 10
        self.n_class = 2
        self.device = 'cuda:1'
        self.seed = 1


class DNDF:
    def __init__(self, forest_size, dataloader, start_temp=None, stop_temp=None, n_in_feature=784):
        self.forests_size = forest_size
        self.dataloader = dataloader
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
            
        instances, instances_y = torch.tensor(np.array(instances).astype(np.float32)), torch.tensor(np.array(instances_y).astype(np.int64))
        instances, instances_y = instances.to(self.opt.device), instances_y.to(self.opt.device)
        return bags, instances, instances_y

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
        bags, instances, instances_y = self.prepare_data(self.dataloader)

        self.init_y = instances_y
        self.neural_forest.train_full(self.optim, instances, instances_y)
        print("ACC:", self.test(instances, instances_y))

        # step 2 - retrain trees substituting labels
        epoch = 0
        temp = self.cooling_fn(epoch)
        while temp > self.stop_temp:

            temp = self.cooling_fn(epoch)

            preds = self.neural_forest.forest.forward(instances)[:, 0].cpu().detach().numpy()
            print(preds[:10])
            confidence = np.abs(0.5-preds)

            # Test: we're treating `preds` as a probability distribution (p*)
            # start = time.time()
            # print("Starting minimization")
            # # probs = minimize(fun=self.calc_p_star,
            # #                  x0=np.full(len(instances), 0.5),
            # #                  args=(preds.detach().cpu().numpy(), temp),
            # #                  bounds=np.tile([tol, 1 - tol], (len(instances), 1)),
            # #                  method='SLSQP')
            # end = time.time()
            # print(f"epoch {epoch}: minimize took {end - start:.3f}s", end=", ")
            start = time.time()
            losses = []
            with tqdm.trange(len(self.neural_forest.forest.trees)) as t:
                for idx in t:
                    for i, (prob, y) in enumerate(zip(preds, instances_y)):
                        instances_y[i] = np.random.choice([0, 1], p=(prob, 1 - prob))

                    highest_idx = np.argmax(confidence)
                    instances_y[highest_idx] = self.init_y[highest_idx]

                    for i in range(2):
                        loss = self.neural_forest.train_tree(idx, instances.to(self.opt.device), instances_y.to(self.opt.device), temp, self.optim)
                    losses.append(loss)
                    t.set_description(f"Loss {np.mean(losses):.3f}")

            epoch += 0.1
            end = time.time()
            print(f"fitting trees took {end - start:.3f}s, acc: {self.test(instances, instances_y):.3f}")

    def predict(self, examples):
        return self.neural_forest.forest.forward(examples)

    def test(self, instances, instances_y):
        instances_y = instances_y.cpu().detach().numpy()
        pred = self.predict(instances).cpu().detach().numpy()
        pred = np.argmax(pred, axis=1)
        return sklearn.metrics.accuracy_score(instances_y, pred)
