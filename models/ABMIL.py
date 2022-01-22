"""
We rewrote and expanded https://github.com/AMLab-Amsterdam/AttentionDeepMIL
according to http://wrap.warwick.ac.uk/77351/7/WRAP_tmi2016_ks.pdf
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as metrics
from tqdm import tqdm


class ABMIL(nn.Module):

    def __init__(self, ds_name, L=128, D=128, K=1, gated=False):
        super().__init__()

        # due to difference in breast and colon cancer patch sizes
        # we need to vary the architecture a little
        if ds_name == "breast_cancer":
            conv1_kernel = 5
            conv2_kernel = 5
        elif ds_name == "colon_cancer":
            conv1_kernel = 4
            conv2_kernel = 3
        else:
            raise NotImplementedError

        self.feature_extractor = nn.Sequential(nn.Conv2d(3, 36, conv1_kernel),
                                               nn.ReLU(),
                                               nn.MaxPool2d(2, 2),
                                               nn.Conv2d(36, 48, conv2_kernel),
                                               nn.ReLU(),
                                               nn.MaxPool2d(2, 2),
                                               nn.Flatten(),
                                               nn.Linear(48 * 5 * 5, 512),
                                               nn.ReLU(),
                                               nn.Dropout(0.2),
                                               nn.Linear(512, L),
                                               nn.ReLU(),
                                               nn.Dropout(0.2))

        self.gated = gated
        if self.gated:
            self.attention_V = nn.Sequential(nn.Linear(L, D),
                                             nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(L, D),
                                             nn.Sigmoid())
            self.attention_weights = nn.Linear(D, K)

        else:
            self.attention = nn.Sequential(nn.Linear(L, D),
                                           nn.Tanh(),
                                           nn.Linear(D, K))

        self.classifier = nn.Sequential(nn.Linear(L * K, 1),
                                        nn.Sigmoid())

        self.bmk = BestModelKeeper()

    def forward(self, x):
        x = x.squeeze()
        x = self.feature_extractor(x)  # NxL

        if self.gated:
            att_v = self.attention_V(x)  # NxD
            att_u = self.attention_U(x)  # NxD
            att = self.attention_weights(att_v * att_u)  # element wise multiplication # NxK

        else:
            att = self.attention(x)  # NxK

        att = torch.transpose(att, 1, 0)  # KxN
        att = F.softmax(att, dim=1)  # softmax over N

        x = torch.mm(att, x)  # KxL

        y_prob = self.classifier(x)
        y_hat = torch.ge(y_prob, 0.5).float()

        return y_prob, y_hat, att

    def train_loop(self, loader, optimizer):
        self.train()

        total_loss = 0.
        total_acc = 0.
        for bag, y_true, _ in tqdm(loader, leave=False):
            optimizer.zero_grad()  # reset gradients
            y_prob, y_hat, _ = self.forward(bag)  # forward pass

            loss = self.objective(y_prob, y_true)  # calculate loss
            loss.backward()  # backward pass
            optimizer.step()  # update params

            total_loss += loss
            total_acc += metrics.accuracy(y_hat, y_true)  # calculate accuracy

        # for convenience return loss and error (instead of accuracy)
        return float(total_loss/len(loader)), 1. - float(total_acc/len(loader))

    def eval_loop(self, loader):
        self.eval()

        total_loss = 0.
        total_acc = 0.
        for bag, y_true, _ in loader:
            y_prob, y_hat, _ = self.forward(bag)  # inference

            total_loss += self.objective(y_prob, y_true)  # calculate loss
            total_acc += metrics.accuracy(y_hat, y_true)  # calculate accuracy

        # for convenience return loss and error (instead of accuracy)
        return float(total_loss/len(loader)), 1. - float(total_acc/len(loader))

    def fit(self, train_loader, test_loader, optimizer, epochs):
        for e in tqdm(range(1, epochs+1), leave=False):
            train_loss, train_error = self.train_loop(train_loader, optimizer)
            test_loss, test_error = self.eval_loop(test_loader)
            tqdm.write(f"epoch {e}:\t"
                       f"train loss/error: {train_loss:.6f}/{train_error:.2f}\t\t"
                       f"test loss/error: {test_loss:.6f}/{test_error:.2f}\t\t"
                       f"loss and error sum: {test_loss + test_error:.4f}")

            # record best (in terms of test accuracy and error sum) model
            self.bmk.routine(e, test_loss + test_error, self)

    def score(self, loader, dict_handle):
        self.eval()

        preds = []
        targets = []
        for bag, y_true, _ in loader:
            _, y_hat, _ = self.forward(bag)  # inference

            preds.append(y_hat)
            targets.append(y_true)

        preds = torch.cat(preds)
        targets = torch.cat(targets)

        dict_handle["accuracy"].append(metrics.accuracy(preds, targets))
        dict_handle["precision"].append(metrics.precision(preds, targets))
        dict_handle["recall"].append(metrics.recall(preds, targets))
        dict_handle["f-score"].append(metrics.f1(preds, targets))
        dict_handle["auc"].append(metrics.auc(preds, targets, reorder=True))

    # negative log bernoulli
    @staticmethod
    def objective(y_prob, target):
        target = torch.tensor(target, dtype=torch.float)
        return -1. * (target * torch.log(y_prob) + (1. - target) * torch.log(1. - y_prob))


class BestModelKeeper:
    def __init__(self, check_after=10):
        self.check_after = check_after
        self.state_dict = None
        self.values = []

    def routine(self, epoch, value, model):

        if epoch > self.check_after:
            self.values.append(value)

            if self.values[-1] == min(self.values):
                self.state_dict = model.state_dict().copy()
                print(f"best model recorded at epoch: {epoch}")
