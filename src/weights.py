import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import matmul_diag_left
from walsh import build_H
from fwht.cpp.fwht import FWHTFunction as FWHT


class WHVISquarePow2Matrix(nn.Module):
    def __init__(self, D, device, lambda_=1e-5):
        super().__init__()
        self.D = D
        self.H = build_H(D, device=device)
        self.device = device
        self.lambda_ = lambda_
        self.padding = 0  # For compatibility with the stacked version

        self.s1 = nn.Parameter(torch.ones(D) * 0.01)
        self.s2 = nn.Parameter(torch.ones(D) * 0.01)
        self.g_mu = nn.Parameter(torch.zeros(D))
        self.g_rho = nn.Parameter(torch.distributions.Uniform(-5, -4).sample((D,)))

    @property
    def g_sigma_sqrt_diagonal(self):
        return F.softplus(self.g_rho)

    @property
    def kl(self):
        # KL divergence from the posterior to the prior, but the prior has zero mean and fully factorized covariance
        # lambda_ * Identity with scalar lambda_.
        kl = 0.5 * (
                self.D * math.log(self.lambda_)
                - torch.sum(2 * torch.log(self.g_sigma_sqrt_diagonal))
                - self.D
                + torch.sum(self.g_sigma_sqrt_diagonal ** 2 / self.lambda_)
                + torch.dot(self.g_mu, self.g_mu) / self.lambda_
        )
        return kl

    def sample(self):
        epsilon = torch.randn(self.D, device=self.device)
        HS2 = self.H * self.s2
        mu_term = matmul_diag_left(self.s1, FWHT.apply(matmul_diag_left(self.g_mu, HS2)))
        sigma_term = matmul_diag_left(self.s1, FWHT.apply(matmul_diag_left(self.g_sigma_sqrt_diagonal * epsilon, HS2)))
        W = mu_term + sigma_term
        return W

    def forward(self, x):
        return F.linear(x, self.sample())


class WHVIStackedMatrix(nn.Module):
    def __init__(self, n_in, n_out, device, lambda_=1e-5):
        """
        WHVI matrix with arbitrary dimensions (i.e. possibly non-square).
        A typical WHVI matrix is square with dimensions D x D where D == 2 ** d for some non-negative integer d.
        This class permits the use of arbitrarily-sized matrices by stacking appropriately-sized square matrices.

        :param n_in: number of input features.
        :param n_out: number of output features.
        :param lambda_: prior variance.
        """
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.device = device
        self.lambda_ = lambda_

        self.D_in, self.D_out, self.padding, self.stack = self.setup_dimensions(n_in, n_out)
        self.weight_matrices = nn.ModuleList([WHVISquarePow2Matrix(self.D_in, device=device, lambda_=lambda_) for _ in range(self.stack)])

    @staticmethod
    def setup_dimensions(D_in, D_out):
        """
        Set up dimensions of a non-square WHVI matrix.

        :param D_in: input dimensionality, i.e. how many features are given to the matrix.
        :param D_out: output dimensionality, i.e. how many features are produced by the matrix.
        :return: 4-tuple of information (D_in_adjusted, D_out_adjusted, padding, stack):
            * D_in_adjusted:  how many columns should the actual matrix have to accommodate the desired number of input
                              features.
            * D_out_adjusted: how many rows should the actual matrix have to accommodate the desired number of output
                              features.
            * padding: how many zeros should be added to an input feature vector to accommodate the adjusted dimensions.
            * stack: how many square matrices need to be stacked together to represent the adjusted non-square matrix.
        """
        next_power = 2 ** math.ceil(math.log(D_in, 2))
        if next_power == 2 * D_in:
            padding = 0
        else:
            padding = next_power - D_in
            D_in = next_power
        stack, remainder = divmod(D_out, D_in)
        if remainder != 0:
            stack += 1
            D_out = D_in * stack
        return D_in, D_out, padding, stack

    @property
    def kl(self):
        return sum(weight.kl for weight in self.weight_matrices)

    def sample(self):
        return torch.cat([weight.sample() for weight in self.weight_matrices])

    def forward(self, x):
        # TODO pre-allocate this vector of zeros if possible
        x_padded = torch.zeros((*x.size()[:-1], self.D_in), device=self.device)  # Add the extra zeros
        x_padded[..., :self.n_in] = x
        output = F.linear(x_padded, self.sample())
        output = output[..., :self.n_out]  # Remove the extra elements
        return output


class WHVIColumnMatrix(nn.Module):
    def __init__(self, n_out, device, lambda_=1e-5, transposed=False):
        super().__init__()
        self.D = n_out
        self.D_adjusted = 2 ** math.ceil(math.log(n_out, 2))
        self.weight_submodule = WHVISquarePow2Matrix(self.D_adjusted, device=device, lambda_=lambda_)
        self.transposed = transposed

    @property
    def kl(self):
        return self.weight_submodule.kl

    def sample(self):
        matrix = torch.reshape(self.weight_submodule.sample(), (-1, 1))[:self.D]
        if self.transposed:
            return matrix.T
        return matrix

    def forward(self, x):
        return F.linear(x, self.sample())
