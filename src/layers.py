import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from walsh import FWHT, FWHT_diag
from utils import matmul_diag_left, matmul_diag_right


# TODO space complexity must be O(D), check FASTFOOD.
# TODO time complexity must be O(DlogD), check FASTFOOD.
# TODO Hx must be computed in O(DlogD) time and O(1) space using the in-place version of FWHT.

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
        self.lambda_ = lambda_

        self.s1 = nn.Parameter(torch.ones(D) * 0.01)  # Diagonal elements of S1
        self.s2 = nn.Parameter(torch.ones(D) * 0.01)  # Diagonal elements of S2
        self.g_mu = nn.Parameter(torch.zeros(D))
        self.g_rho = nn.Parameter(torch.distributions.Uniform(-5, -4).sample((D,)))  # g_sigma_sqrt = softplus(g_rho)

    @property
    def g_sigma_sqrt_diagonal(self):
        return F.softplus(self.g_rho)

    def w_bar(self, u):
        assert u.size() == (self.D,)
        # Is it possible that we can perform FWHT even faster if the input matrix is diagonal?
        return matmul_diag_left(self.s1, FWHT.apply(matmul_diag_left(u, FWHT_diag.apply(self.s2))))

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
        epsilon = torch.randn(self.D)  # Sample independent Gaussian noise
        # Sample W * h according to the local re-parametrization trick. It's also faster to just multiply the
        # diagonal elements with the vector elements than to construct the whole diagonal matrix.
        b = x @ (self.w_bar(self.g_mu) + self.w_bar(self.g_sigma_sqrt_diagonal * epsilon)).T
        return b
