import torch
import torch.nn as nn
import torch.nn.functional as F

from walsh import FWHT, FWHT_diag
from utils import matmul_diag_left, matmul_diag_right


# TODO space complexity must be O(D), check FASTFOOD.
# TODO time complexity must be O(DlogD), check FASTFOOD.
# TODO Hx must be computed in O(DlogD) time and O(1) space using the in-place version of FWHT.

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
    def __init__(self, D):
        """
        WHVI feed forward layer.
        Expects a D-dimensional input and produces a D-dimensional output.

        :param D: number of input (and consequently output) dimensions.
        """
        super().__init__()

        self.D = D

        self.s1 = nn.Parameter(torch.randn(D))  # Diagonal elements of S1
        self.s2 = nn.Parameter(torch.randn(D))  # Diagonal elements of S2
        self.g_mu = nn.Parameter(torch.randn(D))
        self.g_rho = nn.Parameter(torch.randn(D))  # g_sigma_sqrt = softplus(g_rho)

    @property
    def g_sigma_sqrt(self):
        return torch.diag(F.softplus(self.g_rho))

    def w_bar(self, u):
        # Is it possible that we can perform FWHT faster if the input matrix is diagonal?
        return matmul_diag_left(self.s1, FWHT.apply(matmul_diag_left(u, FWHT_diag.apply(self.s2))))

    @property
    def kl(self):
        g_var_post = torch.distributions.Normal(self.g_mu, torch.square(F.softplus(self.g_rho)))
        g_prior = torch.distributions.Normal(0, 1)
        kl = torch.distributions.kl.kl_divergence(g_var_post, g_prior).sum()
        return kl

    def forward(self, x, sample=True):
        S1H = FWHT_diag.apply(self.s1).T
        V = FWHT_diag.apply(self.s2)
        A = torch.cat([(matmul_diag_right(S1H, V[:, i])).T for i in range(self.D)]).T
        if sample:
            epsilon = torch.randn(self.D)  # Sample independent Gaussian noise
            # Sample W * h according to the local re-parametrization trick
            b = x @ self.w_bar(self.g_mu).T + x @ (self.w_bar(self.g_sigma_sqrt @ epsilon)).T
        else:
            W = matmul_diag_right(A, self.g_mu)
            b = F.linear(x, W)
        return b
