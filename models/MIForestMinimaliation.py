import time

import numpy as np
import sklearn
from sklearn import ensemble
from scipy.optimize import minimize
from tqdm import tqdm


class MIForest:
    def __init__(self, forest_size, dataloader, start_temp=None, stop_temp=None):
        self.forests_size = forest_size
        self.dataloader = dataloader
        self.start_temp = start_temp
        self.stop_temp = stop_temp
        self.init_y = []
        self.random_forest = ensemble.RandomForestClassifier(n_estimators=forest_size,
                                                             max_depth=20,
                                                             max_features=25)

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

        self.init_y = instances_y
        self.random_forest.fit(instances, instances_y)

        # step 2 - retrain trees substituting labels

        epoch = 0
        temp = self.cooling_fn(epoch)
        while temp > self.stop_temp:

            temp = self.cooling_fn(epoch)

            preds = self.random_forest.predict_proba(instances)[:, 0]

            start = time.time()
            probs = minimize(fun=self.calc_p_star,
                             x0=np.full(len(instances), 0.5),
                             args=(preds, temp),
                             bounds=np.tile([tol, 1-tol], (len(instances), 1)),
                             method='SLSQP')
            end = time.time()

            print(f"epoch {epoch}: minimize took {end - start:.3f}s", end=", ")

            start = time.time()
            for tree in self.random_forest.estimators_:
                for i, (prob, y) in enumerate(zip(probs.x, instances_y)):
                    instances_y[i] = np.random.choice([0, 1], p=(prob, 1-prob))

                highest_idx = np.argmax(probs)
                instances_y[highest_idx] = self.init_y[highest_idx]

                tree.fit(instances, instances_y)

            epoch += 1
            end = time.time()
            print(f"fitting trees took {end - start:.3f}s")

    def predict(self, examples):
        return self.random_forest.predict(examples)

    def test(self, loader):
        instances, instances_y = self.prepare_data(loader)

        pred = self.predict(instances)
        return sklearn.metrics.accuracy_score(instances_y, pred)
