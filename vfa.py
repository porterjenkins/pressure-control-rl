import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LinearVFA(nn.Module):
    def __init__(self, n_features, n_actions):
        super(LinearVFA, self).__init__()
        self.betas = nn.Linear(n_features, n_actions)
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, X):
        y_hat = self.betas(X)

        return y_hat

    def get_optimizer(self, lr):
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        return optimizer

    def get_loss(self, q_val_curr, q_val_expected):
        # MSE with l2 norm penalty on weights (Lasso)
        err = q_val_curr - q_val_expected
        mse = torch.mean(torch.pow(err, 2))

        return mse


class MlpVFA(LinearVFA):
    def __init__(self, n_features, n_actions):
        super().__init__(n_features, n_actions)
        self.linear_1 = nn.Linear(n_features, 64)
        self.linear_2 = nn.Linear(64, 32)
        self.linear_3 = nn.Linear(32, n_actions)

        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, X):
        h_1 = self.linear_1(X)
        h_2 = self.linear_2(h_1)
        y_hat = self.linear_3(h_2)

        return y_hat


class LstmVFA(nn.Module):
    def __init__(self, seq_size, hidden_dim, n_actions):
        super(LstmVFA, self).__init__()
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions


        self.lstm = nn.LSTM(seq_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_actions)

    def forward(self, X):
        h, (hn, cn) = self.lstm(X)
        y_hat = self.output(hn)

        return y_hat

    def get_optimizer(self, lr):
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        return optimizer

    def get_loss(self, q_val_curr, q_val_expected):
        # MSE with l2 norm penalty on weights (Lasso)
        err = q_val_curr - q_val_expected
        mse = torch.mean(torch.pow(err, 2))

        return mse


if __name__ == '__main__':
    n_features = 8
    n_actions = 4

    x = torch.randn(n_features)
    nn = LinearVFA(n_features, n_actions)
    nn.forward(x)