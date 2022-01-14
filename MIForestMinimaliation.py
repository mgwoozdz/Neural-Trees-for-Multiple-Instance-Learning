import numpy as np
import sklearn.ensemble
from scipy.optimize import minimize

class MIForest:

    def __init__(self, forest_size, bags, bag_labels, start_temp=None, stop_temp=None):
        self.forests_size = forest_size
        self.bags = bags
        self.bag_labels_true = bag_labels
        self.start_temp = start_temp
        self.stop_temp = stop_temp
        self.init_y = []
        self.random_forest = sklearn.ensemble.RandomForestClassifier(n_estimators=forest_size,
                                                                     max_depth=20, max_features=25,
                                                                     random_state=420)

    @staticmethod
    def prepare_data(bags, bags_y):
        # we will process instances directly

        instances = []
        instances_y = []

        for bag, label in zip(bags, bags_y):
            for instance in bag:
                instances.append(instance)
                instances_y.append(label)

        return instances, instances_y

    @staticmethod
    def cooling_fn(epoch, const=0.5):
        return np.exp(-epoch * const)

    def calc_p_star(self, x, examples, t):
        # equation 8 (section 3.1)
        def sigmoid(x):
            return x - 0.5
            #return 1/(1 + np.exp(-x))
        sum = 0
        preds = self.random_forest.predict_proba(examples)
        for index in range(len(x)):
            sum += x[index]*(sigmoid(preds[index][0]) - sigmoid(preds[index][1])) + sigmoid(preds[index][1]) - \
                   t*(x[index]*np.log(x[index]) + (1-x[index])*np.log(1-x[index]))
        return sum
        #return self.random_forest.predict_proba(examples)

    def train(self):

        # step 1 - train random forest using the bag label as label of instances

        instances, instances_y = self.prepare_data(self.bags, self.bag_labels_true)
        self.init_y = instances_y
        self.random_forest.fit(instances, instances_y)

        # step 2 - retrain trees substituting labels

        epoch = 0
        temp = self.cooling_fn(epoch)
        while temp > self.stop_temp:
            temp = self.cooling_fn(epoch)
            tuple_ = np.array([0.5 for i in range(len(instances))])
            bnds = tuple([(0+0.00000001, 1-0.00000001) for i in range(len(instances))])
            probs = minimize(self.calc_p_star, tuple_, (instances, temp), bounds=bnds, method='SLSQP')
            probs = list(map(lambda y: [y, 1-y], probs.x))
            for tree in self.random_forest.estimators_:
                for i, (prob, y) in enumerate(zip(probs, instances_y)):
                    instances_y[i] = np.random.choice([0, 1], p=prob)

                highest_idx = int(np.unravel_index(np.argmax(probs), np.array(probs).shape)[0])
                instances_y[highest_idx] = self.init_y[highest_idx]

                tree.fit(instances, instances_y)

            epoch += 1

    def predict(self, examples):
        return self.random_forest.predict(examples)

    def test(self, examples, labels):
        instances, instances_y = self.prepare_data(examples, labels)

        pred = self.predict(instances)
        return sklearn.metrics.accuracy_score(instances_y, pred)
