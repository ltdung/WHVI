import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import matmul_diag_left, kl_diag_normal, build_H
from fwht.cpp.fwht import FWHT


class WHVISquarePow2Matrix(nn.Module):
    def __init__(self, D, device, lambda_=1e-5, bias=False):
        super().__init__()
        self.D = D
        self.H = build_H(D, device=device)
        self.device = device
        self.lambda_ = lambda_
        self.padding = 0  # For compatibility with the stacked version

        self.bias = nn.Parameter(torch.zeros(1, D)) if bias else None
        self.s1 = nn.Parameter(torch.randn(D))
        self.s2 = nn.Parameter(torch.randn(D))
        self.g_mu = nn.Parameter(torch.zeros(D))
        self.g_rho = nn.Parameter(torch.rand(D) - 10)  # Initialization with Uniform(-10, -9)
        self.FWHT1 = FWHT()  # This is a module
        self.FWHT2 = FWHT()  # This is a module

    @property
    def g_sigma(self):
        # Square roots of the covariance matrix for g (i.e. standard deviations of univariate Normal distributions).
        return F.softplus(self.g_rho)

    @property
    def kl(self):
        return kl_diag_normal(self.g_mu, self.g_sigma, torch.zeros(self.D), torch.ones(self.D) * self.lambda_)

    def sample(self):
        epsilon = torch.randn(self.D, device=self.device)
        g_tilde = self.g_mu + self.g_sigma * epsilon
        W = matmul_diag_left(self.s1) * self.H @ matmul_diag_left(g_tilde, H @ torch.diag(self.s2))
        return W

    def forward(self, x):
        return F.linear(x, self.sample(), self.bias)


class WHVIStackedMatrix(nn.Module):
    def __init__(self, n_in, n_out, device, lambda_=1e-5, bias=False):
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
        self.weight_matrices = nn.ModuleList(
            [WHVISquarePow2Matrix(self.D_in, device=device, lambda_=lambda_) for _ in range(self.stack)])
        self.bias = nn.Parameter(torch.zeros(1, self.D_out)) if bias else None

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
        output = F.linear(x_padded, self.sample(), self.bias)
        output = output[..., :self.n_out]  # Remove the extra elements
        return output


class WHVIColumnMatrix(nn.Module):
    def __init__(self, n_out, device, lambda_=1e-5, bias=False, transposed=False):
        super().__init__()
        self.D = n_out
        self.D_adjusted = 2 ** math.ceil(math.log(n_out, 2))
        self.weight_submodule = WHVISquarePow2Matrix(self.D_adjusted, device=device, lambda_=lambda_)
        self.transposed = transposed
        self.bias = nn.Parameter(torch.zeros(1, 1 if transposed else n_out)) if bias else None

    @property
    def kl(self):
        return self.weight_submodule.kl

    def sample(self):
        matrix = torch.reshape(self.weight_submodule.sample(), (-1, 1))[:self.D]
        if self.transposed:
            return matrix.T
        return matrix

    def forward(self, x):
        return F.linear(x, self.sample(), self.bias)
