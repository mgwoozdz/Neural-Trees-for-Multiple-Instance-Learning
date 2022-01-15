import numpy as np
import sklearn
from sklearn import ensemble
from scipy.optimize import minimize


class MIForest:
    def __init__(self, forest_size, dataloader, start_temp=None, stop_temp=None):
        self.forests_size = forest_size
        self.dataloader = dataloader
        self.start_temp = start_temp
        self.stop_temp = stop_temp
        self.init_y = []
        self.random_forest = ensemble.RandomForestClassifier(n_estimators=forest_size,
                                                             max_depth=20, max_features=25)

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

    def margin(self, p, which):
        res = 2 * p[0] - 1
        if which:
            res *= (-1)
        return res

    def calc_p_star(self, p_hat, preds, t):
        # equation 8 (section 3.1)
        sum = 0
        for index in range(len(p_hat)):
            sum += p_hat[index] * (
                    self.margin(preds[index], False) - self.margin(preds[index], True)) + self.margin(preds[
                                                                                                           index], False) \
                   - t * (p_hat[index] * np.log(p_hat[index]) + (1 - p_hat[index]) * np.log(1 - p_hat[index]))
        return sum
    def train(self):

        # step 1 - train random forest using the bag label as label of instances

        instances, instances_y = self.prepare_data(self.dataloader)

        self.init_y = instances_y
        self.random_forest.fit(instances, instances_y)

        # step 2 - retrain trees substituting labels

        epoch = 0
        temp = self.cooling_fn(epoch)
        while temp > self.stop_temp:
            temp = self.cooling_fn(epoch)
            tuple_ = np.array([0.5 for _ in range(len(instances))])
            preds = self.random_forest.predict_proba(instances)
            bnds = tuple([(0 + 0.00000001, 1 - 0.00000001) for _ in range(len(instances))])
            probs = minimize(self.calc_p_star, tuple_, (preds, temp), bounds=bnds, method='SLSQP')
            probs = list(map(lambda y: [y, 1 - y], probs.x))
            for tree in self.random_forest.estimators_:
                for i, (prob, y) in enumerate(zip(probs, instances_y)):
                    instances_y[i] = np.random.choice([0, 1], p=prob)

                highest_idx = int(np.unravel_index(np.argmax(probs), np.array(probs).shape)[0])
                instances_y[highest_idx] = self.init_y[highest_idx]

                tree.fit(instances, instances_y)

            epoch += 1

    def predict(self, examples):
        return self.random_forest.predict(examples)

    def test(self, loader):
        instances, instances_y = self.prepare_data(loader)

        pred = self.predict(instances)
        return sklearn.metrics.accuracy_score(instances_y, pred)

    # loss function


# preds = self.random_forest.predict_proba(examples)
# for index in range(len(p_hat)):
#     sum += p_hat[index] * (margin(preds[index][0]) - margin(preds[index][1])) + margin(preds[index][1]) - \
#            t * (p_hat[index] * np.log(p_hat[index]) + (1 - p_hat[index]) * np.log(1 - p_hat[index]))
#
# return sum
