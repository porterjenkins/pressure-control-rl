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



    def run_train_job(self, X, y, X_val, y_val, n_epochs, eta, batch_size, lmbda, verbose=True):
        optimizer = self.get_optimizer(eta)
        n_samples = X.shape[0]
        self.loss_seq = list()
        self.loss_seq_val = list()



        for epoch_cntr in range(n_epochs):
            data_idx = np.random.permutation(np.arange(n_samples))

            for step in range(n_samples // batch_size):
                start_idx = step*batch_size
                end_idx = step*batch_size + batch_size
                batch_idx = data_idx[start_idx:end_idx]
                optimizer.zero_grad()

                y_hat = self.forward(X[batch_idx, :])

                loss = self.loss_l2_penalty(y_hat, y[batch_idx], lmbda)
                loss.backward()
                optimizer.step()

            y_hat_val = self.forward(X_val)
            loss_val = self.loss_l2_penalty(y_hat_val, y_val, lmbda)

            if verbose:
                print("Epoch: {}, Train Loss: {:.4f}, Val. loss: {:.4f}".format(epoch_cntr, loss.item(), loss_val.item()))
            self.loss_seq.append(loss.item())
            self.loss_seq_val.append(loss_val.item())


if __name__ == '__main__':
    n_features = 8
    n_actions = 4

    x = torch.randn(n_features)
    nn = LinearVFA(n_features, n_actions)
    nn.forward(x)