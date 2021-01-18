import random

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from walsh import build_H, FWHT
from utils import matmul_diag_left


class WHVI:
    def __init__(self):
        """
        Abstract class for WHVI layers.
        """
        self.kl = 0.0


class WHVILinear(nn.Module, WHVI):
    def __init__(self, D, lambda_=0.00001):
        """
        WHVI feed forward layer.
        Expects a D-dimensional input and produces a D-dimensional output.

        :param D: number of input (and consequently output) dimensions.
        :param lambda_: prior variance.
        """
        super().__init__()

        self.D = D
        self.H = build_H(D)
        self.lambda_ = lambda_
        # self.epsilon_block = torch.randn(D * 30)

        self.s1 = nn.Parameter(torch.ones(D) * 0.01)  # Diagonal elements of S1 - what is a good initialization? TODO
        self.s2 = nn.Parameter(torch.ones(D) * 0.01)  # Diagonal elements of S2 - what is a good initialization? TODO
        self.g_mu = nn.Parameter(torch.zeros(D))
        self.g_rho = nn.Parameter(torch.distributions.Uniform(-5, -4).sample((D,)))  # g_sigma_sqrt = softplus(g_rho)

    @property
    def g_sigma_sqrt_diagonal(self):
        return F.softplus(self.g_rho)

    def w_bar(self, u):
        assert u.size() == (self.D,)
        # Is it possible that we can perform FWHT even faster if the input matrix is diagonal?
        return matmul_diag_left(self.s1, FWHT.apply(matmul_diag_left(u, self.H * self.s2)))

    @property
    def kl(self):
        # KL divergence from the posterior to the prior, but the prior has zero mean and fully factorized covariance
        # lambda_ * Identity.
        g_sigma_diagonal = self.g_sigma_sqrt_diagonal ** 2
        kl = 0.5 * (
                self.D * math.log(self.lambda_)
                - torch.sum(torch.log(g_sigma_diagonal))
                - self.D
                + torch.sum(g_sigma_diagonal / self.lambda_)
                + torch.dot(self.g_mu, self.g_mu) / self.lambda_
        )
        return kl

    def forward(self, x):
        # Sample Gaussian noise with the gaussian block trick.
        # epsilon = self.epsilon_block[torch.tensor(random.sample(range(len(self.epsilon_block)), self.D))]
        epsilon = torch.randn(self.D)  # Sample without the trick

        # Sample W * h according to the local re-parametrization trick. It's also faster to just multiply the
        # diagonal elements with the vector elements than to construct the whole diagonal matrix.
        b = x @ (self.w_bar(self.g_mu) + self.w_bar(self.g_sigma_sqrt_diagonal * epsilon)).T
        return b
