import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F


class NeuralForest():
    def __init__(self, forest, opt):
        self.forest = forest
        # self.optimizer = None
        # self.criterion = None
        self.loss = None
        self.opt = opt
        if self.opt.cuda:
            for idx in range(len(self.forest.trees)):
                self.forest.trees[idx].feature_mask = self.forest.trees[idx].feature_mask.cuda()
                self.forest.trees[idx].decision = self.forest.trees[idx].decision.cuda()

    def get_params(self):
        params = [p for p in self.forest.parameters() if p.requires_grad]
        return params

    def train_tree(self, index, feature_batches, target_batches, optim=None):
        tree = self.forest.trees[index]
        mu_batches = []
        for feats, target in zip(feature_batches, target_batches):
            mu = tree(feats)  # [batch_size,n_leaf]
            mu_batches.append(mu)
            if optim is not None:
                output = tree.cal_prob(mu, tree.get_pi().cuda())
                loss = F.nll_loss(torch.log(output), target.argmax(1).cuda())
                loss.backward()
                optim.step()

        with torch.no_grad():
            for _ in range(20):
                new_pi = torch.zeros((tree.n_leaf, tree.n_class))  # Tensor [n_leaf,n_class]
                if self.opt.cuda:
                    new_pi = new_pi.cuda()
                for mu, target in zip(mu_batches, target_batches):
                    pi = tree.get_pi()  # [n_leaf,n_class]
                    prob = tree.cal_prob(mu.cuda(), pi.cuda())  # [batch_size,n_class]

                    # Variable to Tensor
                    pi = pi.data
                    prob = prob.data
                    mu = mu.data

                    _target = target.unsqueeze(1).cuda()  # [batch_size,1,n_class]
                    _pi = pi.unsqueeze(0).cuda()  # [1,n_leaf,n_class]
                    _mu = mu.unsqueeze(2).cuda()  # [batch_size,n_leaf,1]
                    _prob = torch.clamp(prob.unsqueeze(1), min=1e-6,
                                        max=1.).cuda()  # [batch_size,1,n_class]

                    _new_pi = torch.mul(torch.mul(_target, _pi),
                                        _mu) / _prob  # [batch_size,n_leaf,n_class]
                    new_pi += torch.sum(_new_pi, dim=0)

                new_pi = F.softmax(Variable(new_pi), dim=1).data
                tree.update_pi(new_pi)
                if self.opt.cuda:
                    tree.pi = tree.pi.cuda()

    def train(self, optim, db):
        for epoch in range(1, self.opt.epochs + 1):
            # Update \Pi
            self.forest.train()
            if not self.opt.jointly_training:
                print("Epoch %d : Two Stage Learning - Update PI" % (epoch))
                # prepare feats
                cls_onehot = torch.eye(self.opt.n_class)
                feat_batches = []
                target_batches = []
                train_loader = torch.utils.data.DataLoader(db['train'],
                                                           batch_size=self.opt.batch_size,
                                                           shuffle=True)
                # with torch.no_grad():
                for batch_idx, (data, target) in enumerate(train_loader):
                    if self.opt.cuda:
                        data, target, cls_onehot = data.cuda(), target.cuda(), cls_onehot.cuda()
                    data = Variable(data)
                    # Get feats
                    feats = data.view(data.size()[0], -1)
                    feat_batches.append(feats)
                    target_batches.append(cls_onehot[target])

                # Update \Pi for each tree
                for idx in range(len(self.forest.trees)):
                    self.train_tree(idx, feat_batches, target_batches)

            self.train_full(optim, db)

            # Eval
            self.forest.eval()
            test_loss = 0
            correct = 0
            test_loader = torch.utils.data.DataLoader(db['eval'], batch_size=self.opt.batch_size,
                                                      shuffle=True)
            with torch.no_grad():
                for data, target in test_loader:
                    if self.opt.cuda:
                        data, target = data.cuda(), target.cuda()

                    data = data.view(data.size()[0], -1)
                    data, target = Variable(data), Variable(target)
                    output = self.forest(data)
                    test_loss += F.nll_loss(torch.log(output), target,
                                            size_average=False).item()  # sum up batch loss
                    pred = output.data.max(1, keepdim=True)[
                        1]  # get the index of the max log-probability
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

                test_loss /= len(test_loader.dataset)
                print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
                    test_loss, correct, len(test_loader.dataset),
                    correct / len(test_loader.dataset)))

    def train_full(self, optim, instances, instances_y):
        self.forest.train()
        # train_loader = torch.utils.data.DataLoader(db['train'], batch_size=self.opt.batch_size,
        #                                            shuffle=True)

        for data, target in zip(instances, instances_y):
            data = torch.tensor(data)
            target = torch.tensor(target)

            if self.opt.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            data = data.view(data.size()[0], -1)
            optim.zero_grad()
            output = self.forest(data)
            loss = F.nll_loss(torch.log(output), target)
            loss.backward()
            optim.step()

        return loss.item()
