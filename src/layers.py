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


class BasicWHVILinear(nn.Module):
    def __init__(self, D, activation=F.relu):
        """
        Bayesian feed forward layer.
        Uses WHVI to estimate the mean and variance of its weights.
        Expects a D-dimensional input and produces a D-dimensional output.
        """
        super().__init__()

        self.D = D
        self.H = build_H(D)
        self.act = activation

        self.s1 = nn.Parameter(torch.rand(D))  # Diagonal elements of S1
        self.s2 = nn.Parameter(torch.rand(D))  # Diagonal elements of S2

        self.q_mu = nn.Parameter(torch.randn(D))
        self.q_factor_lower = nn.Parameter(torch.tril(torch.randn(D, D)))  # This is probably not ideal for sampling

    @property
    def A(self):
        S1H = torch.diag(self.s1) @ self.H
        # TODO use FWHT to compute this faster
        HS2 = self.H @ torch.diag(self.s2)
        A = torch.cat([S1H @ torch.diag(HS2[:, i]) for i in range(self.D)], 1).T
        return A

    @property
    def q_Sigma(self):
        q_factor = self.q_factor_lower + self.q_factor_lower.T
        for i in range(self.D):
            q_factor[i, i] = self.q_factor_lower[i, i]
        return q_factor @ q_factor.T

    def sample_W(self):
        q = torch.distributions.MultivariateNormal(self.q_mu, self.q_Sigma)  # Variational posterior
        g = q.sample()
        vect_W = self.A @ g
        W = torch.reshape(vect_W, (self.D, self.D))
        return W

    def forward(self, x):
        return self.act(F.linear(x, self.sample_W(), bias=None))
