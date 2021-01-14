import torch
import torch.nn as nn
import torch.nn.functional as F

from walsh import FWHT, FWHT_diag
from utils import matmul_diag_left, matmul_diag_right


# TODO space complexity must be O(D), check FASTFOOD.
# TODO time complexity must be O(DlogD), check FASTFOOD.
# TODO Hx must be computed in O(DlogD) time and O(1) space using the in-place version of FWHT.


class WHVILinear(nn.Module):
    def __init__(self, D):
        """
        WHVI feed forward layer.
        Expects a D-dimensional input and produces a D-dimensional output.

        :param D: number of input (and consequently output) dimensions.
        """
        super().__init__()

        self.D = D

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
        g_var_post = torch.distributions.Normal(self.g_mu, self.g_sigma_sqrt_diagonal)
        g_prior = torch.distributions.Normal(0, 10 ** (-5))
        kl = torch.distributions.kl.kl_divergence(g_var_post, g_prior).sum()
        return kl

    def forward(self, x, sample=True):
        S1H = FWHT_diag.apply(self.s1).T
        V = FWHT_diag.apply(self.s2)
        A = torch.cat([(matmul_diag_right(S1H, V[:, i])).T for i in range(self.D)]).T
        if sample:
            epsilon = torch.randn(self.D)  # Sample independent Gaussian noise
            # Sample W * h according to the local re-parametrization trick. It's also faster to just multiply the
            # diagonal elements with the vector elements than to construct the whole diagonal matrix.
            b = x @ (self.w_bar(self.g_mu) + self.w_bar(self.g_sigma_sqrt_diagonal * epsilon)).T
        else:
            W = matmul_diag_right(A, self.g_mu)
            b = F.linear(x, W)
        return b
