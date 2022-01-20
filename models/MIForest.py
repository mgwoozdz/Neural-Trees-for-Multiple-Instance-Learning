import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class MIForest:
    def __init__(self, forest, optim=None,  feature_layer=None,  dataloader=None, start_temp=None, stop_temp=None):
        self.forest = forest
        self.dataloader = dataloader
        self.start_temp = start_temp
        self.stop_temp = stop_temp
        self.init_y = []
        self.feature_layer = feature_layer
        self.optim = optim

    @staticmethod
    def prepare_data(bags, labels):
        # we will process instances directly
        instances = []
        instances_y = []

        for bag, label in zip(bags, labels):
            for instance in bag:
                instance = instance.detach().cpu()
                instances.append(np.array(instance).flatten())
                instances_y.append(label)

        instances = torch.tensor(instances, dtype=torch.float32)
        instances_y = torch.tensor(instances_y, dtype=torch.float32)

        return instances, instances_y

    @staticmethod
    def cooling_fn(epoch, const=0.5):
        return np.exp(-epoch * const)

    def calc_p_star(self, x):
        self.forest.eval()
        return self.forest(x)

    def train_forest(self, instance, label):
        self.forest.train()
        for x, y in zip(instance, label):
            self.optim.zero_grad()
            output = self.forest(x.view(1, -1))
            y = torch.tensor(y, dtype=torch.long)
            loss = F.nll_loss(torch.log(output), y.view(-1))
            loss.backward()
            self.optim.step()

    def train(self, bags, labels):
        instances, instances_y = self.prepare_data(bags, labels)

        # copy correct labels
        self.init_y = instances_y[:]

        # train forest
        self.train_forest(instances, instances_y)

        epoch = 0
        temp = self.cooling_fn(epoch)

        while temp > self.stop_temp:
            # get probabilities from all instances (x)
            probs = self.calc_p_star(instances)
            probs = probs.detach()

            # set random label
            for i, prob in enumerate(probs):
                instances_y[i] = np.random.choice([0, 1], p=prob.detach())

            # set bag label for x with highest prop
            highest_idx = int(np.unravel_index(np.argmax(probs), np.array(probs).shape)[0])
            instances_y[highest_idx] = self.init_y[highest_idx]

            # retrain forest
            self.train_forest(instances, instances_y)

            print(epoch)
            epoch += 1
            temp = self.cooling_fn(epoch)

    def predict(self, examples):
        self.forest.eval()
        return self.forest(examples)

    def test(self, bags, labels):
        instances, instances_y = self.prepare_data(bags, labels)

        test_loss = 0
        correct = 0

        self.forest.eval()

        with torch.no_grad():
            for x, y in zip(instances, instances_y):
                data, target = Variable(x), Variable(y)
                target = torch.tensor(target, dtype=torch.long)
                output = self.forest(data.view(1, -1))
                test_loss += F.nll_loss(torch.log(output), target.view(-1), size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            test_loss /= len(instances)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
                test_loss, correct, len(instances),
                correct / len(instances)))

            # pred = self.predict(instances)
            return correct / len(instances)
