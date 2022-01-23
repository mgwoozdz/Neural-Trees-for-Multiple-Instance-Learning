import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F


class Tree(nn.Module):
    def __init__(self, depth, n_in_feature, used_feature_rate, n_class, jointly_training=True):
        super(Tree, self).__init__()
        self.device = 'cpu'
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.n_class = n_class
        self.jointly_training = jointly_training

        # used features in this tree
        n_used_feature = int(n_in_feature * used_feature_rate)
        onehot = np.eye(n_in_feature)
        using_idx = np.random.choice(np.arange(n_in_feature), n_used_feature, replace=False)
        self.feature_mask = onehot[using_idx].T
        self.feature_mask = Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor),
                                      requires_grad=False).cuda()
        # leaf label distribution
        if jointly_training:
            self.pi = np.random.rand(self.n_leaf, n_class)
            self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor),
                                requires_grad=True)
        else:
            self.pi = np.ones((self.n_leaf, n_class)) / n_class
            self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor),
                                requires_grad=False)

        # decision
        self.decision = self.get_network(n_used_feature)
        # self.target_decision = self.get_network(n_used_feature)

        self.optimizer = torch.optim.Adam(self.decision.parameters(), lr=3e-4)

    def get_network(self, n_used_feature):
        return nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(n_used_feature, self.n_leaf)),
            ('sigmoid', nn.Sigmoid()),
        ])).cuda()

    def soft_update_weight(self, tau=0.99):
        for target_param, param in zip(self.target_decision.parameters(), self.decision.parameters()):
            param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def forward(self, x, target=False):
        """
        :param x(Variable): [batch_size,n_features]
        :return: route probability (Variable): [batch_size,n_leaf]
        """
        # if x.is_cuda and not self.feature_mask.is_cuda:
        #     self.feature_mask = self.feature_mask.cuda()
        feats = torch.mm(x, self.feature_mask)  # ->[batch_size,n_used_feature]
        decision = self.decision(feats)  # ->[batch_size,n_leaf]
        # if not target:
        #     decision = self.decision(feats)
        # else:
        #     decision = self.target_decision(feats)

        decision = torch.unsqueeze(decision, dim=2)
        decision_comp = 1 - decision
        decision = torch.cat((decision, decision_comp), dim=2)  # -> [batch_size,n_leaf,2]

        # compute route probability
        # note: we do not use decision[:,0]
        batch_size = x.size()[0]
        _mu = Variable(x.data.new(batch_size, 1, 1).fill_(1.))
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _decision = decision[:, begin_idx:end_idx, :]  # -> [batch_size,2**n_layer,2]
            _mu = _mu * _decision  # -> [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer + 1)

        mu = _mu.view(batch_size, self.n_leaf)

        return mu

    def get_pi(self):
        if self.jointly_training:
            return F.softmax(self.pi, dim=-1)
        else:
            return self.pi

    def cal_prob(self, mu, pi):
        """
        :param mu [batch_size,n_leaf]
        :param pi [n_leaf,n_class]
        :return: label probability [batch_size,n_class]
        """
        p = torch.mm(mu, pi.to(self.device))
        return p

    def update_pi(self, new_pi):
        self.pi.data = new_pi
