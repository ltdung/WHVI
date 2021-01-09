import torch
import torch.nn as nn
import torch.nn.functional as F


def build_H(D, scale=True):
    assert (D & (D >> 1)) == 0 and D > 0, "Error: D must be a power of two."
    H = build_H_recursive(D)
    if scale:
        H = (D ** (-1 / 2)) * H  # Make H orthonormal
    return H


def build_H_recursive(D):
    if D == 2:
        return torch.tensor(data=[[1., 1.], [1., -1.]])
    half_H = build_H_recursive(D // 2)  # Division by 2
    return torch.cat([
        torch.cat([half_H, half_H], dim=1),
        torch.cat([half_H, -half_H], dim=1)
    ], dim=0)


class WHVILinear(nn.Module):
    def __init__(self, D, activation=F.relu):
        """
        WHVI feed forward layer.
        Expects a D-dimensional input and produces a D-dimensional output.

        :param D: number of input (and consequently output) dimensions.
        :param activation: torch activation function (default: ReLU).
        """
        super().__init__()

        self.D = D
        self.H = build_H(D)
        self.act = activation

        self.s1 = nn.Parameter(torch.randn(D))  # Diagonal elements of S1
        self.s2 = nn.Parameter(torch.randn(D))  # Diagonal elements of S2
        self.mu = nn.Parameter(torch.randn(D))
        self.Rho = nn.Parameter(torch.randn((D, D)))  # Sigma_sqrt = softplus(Rho) @ softplus(Rho).T

    def w_bar(self, u):
        return torch.diag(self.s1) @ self.H @ torch.diag(u) @ self.H @ torch.diag(self.s2)  # TODO use FWHT

    def sample_b(self, h):
        # Sample W * h according to the local re-parametrization trick.
        sigma_sqrt_factor = F.softplus(self.Rho)  # Make all elements non-negative
        sigma_sqrt = sigma_sqrt_factor @ sigma_sqrt_factor.T  # Make the matrix positive-definite
        epsilon = torch.randn(self.D)  # Sample independent Gaussian noise.
        return self.w_bar(self.mu) @ h + self.w_bar(sigma_sqrt @ epsilon) @ h

    def forward(self, x):
        b = self.sample_b(x.T)
        return self.act(b)


if __name__ == '__main__':
    torch.manual_seed(0)
    layer = WHVILinear(2)
    inputs = torch.tensor(data=[
        [1., 0.],
        [0., 1.],
        [0., 2.],
        [0., 3.],
        [4., 0.],
    ])
    targets = torch.tensor(data=[
        [5., 0.],
        [0., 5.],
        [0., 10.],
        [0., 15.],
        [20., 0.],
    ])
    outputs = layer(inputs)
    print(outputs)
