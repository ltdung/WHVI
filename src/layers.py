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
    def __init__(self, D, activation=F.relu, w_prior=None):
        """
        WHVI feed forward layer.
        Expects a D-dimensional input and produces a D-dimensional output.

        :param D: number of input (and consequently output) dimensions.
        :param activation: torch activation function (default: ReLU).
        :param w_prior: prior distribution for the weights (default: diagonal multivariate standard normal).
        :type w_prior: torch.distributions.distribution.Distribution.
        """
        super().__init__()

        self.D = D
        self.H = build_H(D)
        self.act = activation

        self.s1 = nn.Parameter(torch.rand(D))  # Diagonal elements of S1
        self.s2 = nn.Parameter(torch.rand(D))  # Diagonal elements of S2

        self.q_mu = nn.Parameter(torch.randn(D))
        self.q_factor_lower = nn.Parameter(torch.tril(torch.randn(D, D)))  # This is probably not ideal for sampling
        # TODO self.q_factor_lower seems wrong, because the paper says the complexity is linear in D

        if w_prior is None:
            self.w_prior = torch.distributions.Normal(0, 1)
        else:
            self.w_prior = w_prior
        self.log_prior = 0.0
        self.log_var_posterior = 0.0

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
        log_prior = self.w_prior.log_prob(W).sum()
        log_var_posterior = q.log_prob(g).sum()  # TODO this seems wrong
        return W, log_prior, log_var_posterior

    def forward(self, x):
        W, log_prior, log_var_posterior = self.sample_W()
        self.log_prior = log_prior
        self.log_var_posterior = log_var_posterior
        return self.act(F.linear(x, W, bias=None))
