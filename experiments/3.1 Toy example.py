import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from layers import WHVILinear

torch.manual_seed(0)  # Seed for reproducibility
n = 128  # Number of sampled inputs

uniform = torch.distributions.Uniform(-1, 2)
x = uniform.sample((n,))

# It *seems* that [-1 + 3/5 * 3, -1 + 3/5 * 4] is the predefined interval for removing 20% of inputs.
x = x[torch.where((x < -1 + 3 / 5 * 3) | (x > -1 + 3 / 5 * 4))]

# It *seems* that y = f(x) + eps where eps ~ N(0, exp(-3)). The authors do not explicitly say what f is.
# For this reason, we define a similar polynomial and see if we observe a similar pattern.
# The polynomial coefficients are chosen based on observed extrema in the plot.
xs_poly = [-2.0, -1.5, -0.8, 0.0, 0.5, 1.4, 2.0, 2.7, 3.0]
ys_poly = [1.2, 1.5, 2.0, 0.5, -0.5, 1.2, 0.0, 1.0, 1.3]
vmat = np.vander(xs_poly, len(xs_poly))
coef = np.linalg.solve(vmat, ys_poly)


def f(x):
    return torch.from_numpy(np.polyval(coef, x))


y = f(x) + torch.distributions.Normal(0, np.sqrt(np.exp(-3))).sample((len(x),))

plt.figure()
xs_ls = torch.linspace(-2, 3, 1000)
plt.title('Data and function for experiment 3.1 (Toy example)')
plt.plot(xs_ls, f(xs_ls), label='True function')
plt.scatter(x, y, label='Function samples')
plt.scatter(xs_poly, ys_poly, label='Polynomial keypoints')
plt.ylim(-1, 2.5)
plt.xlim(-2, 3)
plt.legend()
plt.show()


class ToyNetwork(nn.Module):
    def __init__(self, n_in=1, n_out=1, D=128, loss_function=F.mse_loss):
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
        self.l3 = WHVILinear(D)
        self.l4 = nn.Linear(D, n_out)

    def forward(self, x):
        x = torch.cos(self.l1(x))
        x = torch.cos(self.l2(x))
        x = torch.cos(self.l3(x))
        x = self.l4(x)
        return x

    def loss(self, X, y, n_samples=10):
        assert n_samples > 0

        nll_samples = torch.zeros(n_samples)
        for i in range(n_samples):
            y_hat = self(X)
            nll = self.loss_function(y_hat, y)
            nll_samples[i] = nll

        mean_nll = nll_samples.mean()
        kl = self.l2.kl + self.l3.kl  # Take KL terms of all WHVI layers and sum them
        print(f'NLL: {float(mean_nll):.2f}, KL: {float(kl):.2f}', end=', ')
        return mean_nll + kl


train_x = torch.reshape(x, (-1, 1))
train_y = torch.reshape(y, (-1, 1))
net = ToyNetwork()
net.train()
optimizer = optim.Adam(net.parameters(), lr=0.15)  # Optimizer not given in paper
for epoch in range(200):  # Number of epochs not given in paper
    loss = net.loss(train_x, train_y, n_samples=10)  # Number of samples not given in paper
    loss.backward()
    # print(f'Epoch {epoch + 1}, loss = {float(loss):.3f}')
    print(f'MVar2: {float(F.softplus(net.l2.g_rho).mean()):.3f}, MVar3: {float(F.softplus(net.l3.g_rho).mean()):.3f}')
    # print(net.l2.s1[:5])
    optimizer.step()
    net.zero_grad()

plt.figure()
# plt.title('Data and function for experiment 3.1 (Toy example)')
plt.ylim(-1, 2.5)
plt.xlim(-2, 3)
test_x = torch.reshape(torch.linspace(-2, 3, 1000), (-1, 1))
for _ in range(100):
    out_y = net(test_x)
    plt.plot(test_x.numpy().ravel(), out_y.detach().numpy().ravel(), c='b', alpha=0.05)
plt.plot(xs_ls, f(xs_ls), label='True function')
plt.scatter(x, y, label='Function samples')
plt.legend()
plt.show()
