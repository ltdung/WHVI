import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from layers import WHVILinear


class WHVIRegression(nn.Module):
    def __init__(self, n_in, n_out, D, loss_function=F.mse_loss):
        """
        Regression feed-forward neural network with a single WHVI layer.

        :param n_in: input dimensionality.
        :param n_out: output dimensionality.
        :param D: number of WHVI weight matrix rows/columns.
        """
        super().__init__()
        assert (D & (D >> 1)) == 0 and D > 0
        self.loss_function = loss_function

        self.l1 = nn.Linear(n_in, D)
        self.l2 = WHVILinear(D)
        self.l3 = nn.Linear(D, n_out)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    def loss(self, X, y, n_samples=10):
        assert n_samples > 0

        nll_samples = torch.zeros(n_samples)
        kl_samples = torch.zeros(n_samples)
        for i in range(n_samples):
            y_hat = self(X)
            nll = self.loss_function(y_hat, y)
            kl = self.l2.kl  # Take KL terms of all WHVI layers
            nll_samples[i] = nll
            kl_samples[i] = kl

        mean_nll = nll_samples.mean()
        mean_kl = kl_samples.mean()
        return mean_nll + mean_kl


if __name__ == '__main__':

    torch.manual_seed(0)
    inputs = torch.tensor(data=[
        [1., 0.],
        [0., 1.],
        [0., 1.2],
        [0., 1.3],
        [1.4, 0.],
    ])
    targets = torch.reshape((inputs ** 3 - torch.exp(inputs * 2)).sum(dim=1), (-1, 1))  # A toy function

    net = WHVIRegression(n_in=2, n_out=1, D=8, loss_function=F.mse_loss)
    net.train()

    optimizer = optim.Adam(net.parameters())
    loss_history = []
    for epoch in range(500):
        loss = net.loss(inputs, targets)
        loss_history.append(float(loss))
        loss.backward()
        optimizer.step()
        net.zero_grad()

    import matplotlib.pyplot as plt
    plt.plot(loss_history)
    plt.show()
