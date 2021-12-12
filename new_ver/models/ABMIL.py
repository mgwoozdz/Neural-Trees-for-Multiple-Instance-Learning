import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):

    def __init__(self, L=500, D=128, K=1, gated=False):
        super().__init__()

        self.gated = gated

        self.feature_extractor_part1 = nn.Sequential(nn.Conv2d(3, 20, kernel_size=5), nn.ReLU(),
                                                     nn.MaxPool2d(2, stride=2),
                                                     nn.Conv2d(20, 50, kernel_size=5), nn.ReLU(),
                                                     nn.MaxPool2d(2, stride=2))

        self.feature_extractor_part2 = nn.Sequential(nn.Linear(50 * 4 * 4, L), nn.ReLU())

        if self.gated:
            self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
            self.attention_weights = nn.Linear(D, K)

        else:
            self.attention = nn.Sequential(nn.Linear(L, D), nn.Tanh(),
                                           nn.Linear(D, K))

        self.classifier = nn.Sequential(nn.Linear(L * K, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(0)

        x = self.feature_extractor_part1(x)
        x = x.view(-1, 50 * 4 * 4)
        x = self.feature_extractor_part2(x)  # NxL

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
