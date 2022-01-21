"""
Implementation of model described by algorithm 1 from
https://link.springer.com/content/pdf/10.1007%2F978-3-642-15567-3_3.pdf.
"""

import time
import numpy as np
from sklearn import ensemble
from scipy.optimize import minimize


class MIForest:

    def __init__(self, n_estimators=50, max_depth=20, cooling_fn_const=0.5, stop_temp=0.005):
        self.forest = ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        self.cooling_fn = lambda epoch: np.exp(-epoch * cooling_fn_const)
        self.stop_temp = stop_temp

    def fit(self, instances, labels, tol=1e-6):

        # step 1 - pretrain random forest on ground true labels (derived form bags)
        self.forest.fit(instances, labels)

        # step 2 - retrain trees substituting labels
        sub_labels = labels.copy()

        epoch = 0
        temp = self.cooling_fn(epoch)
        while temp > self.stop_temp:

            preds = self.forest.predict_proba(instances)[:, 0]

            start = time.time()
            probs = minimize(fun=self.objective_fn,
                             x0=np.full(len(instances), 0.5),
                             args=(preds, temp),
                             bounds=np.tile([tol, 1-tol], (len(instances), 1)),
                             method='SLSQP')
            end = time.time()
            print(f"epoch {epoch}: minimize took {(end - start):.3f}s", end=", ")

            start = time.time()
            for tree in self.forest.estimators_:
                # TODO: try replacing loop with numpy op
                for i, (prob, y) in enumerate(zip(probs.x, sub_labels)):
                    sub_labels[i] = np.random.choice([0, 1], p=(prob, 1 - prob))

                highest_idx = np.argmax(probs)
                sub_labels[highest_idx] = labels[highest_idx]

                tree.fit(instances, sub_labels)
            end = time.time()
            print(f"fitting trees took {end - start:.3f}s")

            epoch += 1
            temp = self.cooling_fn(epoch)

    def predict(self, instances):
        # TODO: fix bug of reversed labels (check loss term in objective_fn)
        return 1-self.forest.predict(instances)

    @staticmethod
    def objective_fn(p_hat, preds, temp):
        # equation 8 (section 3.1)
        loss_term = p_hat * (4 * preds - 2) + 2 * preds - 1
        temp_term = p_hat * np.log(p_hat) + (1 - p_hat) * np.log(1 - p_hat)
        return np.sum(loss_term + temp * temp_term)

    @staticmethod
    def bags_to_instances(bags, labels):
        instances = []
        instances_y = []
        # TODO: use numpy broadcasting instead of double loop
        for bag, label in zip(bags, labels):
            for instance in bag:
                instances.append(instance)
                instances_y.append(label)

        return np.array(instances), np.array(instances_y)
