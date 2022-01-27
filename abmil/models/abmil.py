from pathlib import Path

import torch as t
import torch.nn.functional as F
from torch import nn

SAVE_PATH = Path(__file__).parent / "abmil.pth"


class AttentionBasedMIL(nn.Module):
    def __init__(self, return_repr=False):
        super().__init__()

        self.return_repr = return_repr

        self.ATTENTION_IN_DIM = 500
        self.ATTENTION_OUT_DIM = 128

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 36, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(36, 48, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.__xavier_initialize_uniform_and_bias_zero(self.feature_extractor, [0, 3])

        self.fc = nn.Sequential(
            nn.Linear(48 * 5 * 5, self.ATTENTION_IN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.ATTENTION_IN_DIM, self.ATTENTION_IN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.__xavier_initialize_uniform_and_bias_zero(self.fc, [0, 3])

        self.attention = nn.Sequential(
            nn.Linear(self.ATTENTION_IN_DIM, self.ATTENTION_OUT_DIM),
            nn.Tanh(),
            nn.Linear(self.ATTENTION_OUT_DIM, 1),
        )

        self.__xavier_initialize_uniform_and_bias_zero(self.attention, [0, 2])

        self.classifier = nn.Sequential(
            nn.Linear(self.ATTENTION_IN_DIM, 1), nn.Sigmoid(),
        )

        self.__xavier_initialize_uniform_and_bias_zero(self.classifier, [0])

    @staticmethod
    def __xavier_initialize_uniform_and_bias_zero(sequential, ids):
        for param_id in ids:
            nn.init.xavier_uniform_(sequential[param_id].weight)
            sequential[param_id].bias.data.zero_()

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor(x)

        H = H.view(-1, 48 * 5 * 5)
        H = self.fc(H)

        if self.return_repr:
            return H

        A = self.attention(H)
        A = t.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        M = t.mm(A, H)

        Y_prob = self.classifier(M)
        Y_hat = t.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1.0 - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = t.clamp(Y_prob, min=1e-5, max=1.0 - 1e-5)
        neg_log_likelihood = -1.0 * (
            Y * t.log(Y_prob) + (1.0 - Y) * t.log(1.0 - Y_prob)
        )

        return neg_log_likelihood, A
