import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import matmul_diag_left, kl_diag_normal

from cuda.fwht import FWHTFunction as fwht_cuda
from cpp.fwht import FWHTFunction as fwht_cpp
from python.fwht import WHT_matmul as wht_matmul


class WHVISquarePow2Matrix(nn.Module):
    def __init__(self, D, lambda_=1e-5, bias=False):
        """
        Create a square WHVI matrix of size (D, D) where D is a power of 2.

        :param int D: number of rows/columns for the matrix. Must be a power of two.
        :param float lambda_: prior variance.
        :param boolean bias: if True, include a bias in the linear operation.
        """
        super().__init__()
        self.D = D
        self.lambda_ = lambda_
        self.padding = 0  # For compatibility with the stacked version
        self.wht_slow = wht_matmul()  # In case we need regular matmul, but this does not create H otherwise.

        self.bias = nn.Parameter(torch.zeros(1, D)) if bias else None
        self.s1 = nn.Parameter(torch.randn(D))
        self.s2 = nn.Parameter(torch.randn(D))
        self.g_mu = nn.Parameter(torch.zeros(D))
        self.g_rho = nn.Parameter(torch.rand(D) - 3)

    def fwht(self, x):
        if x.device.type == "cuda":
            return fwht_cuda.apply(x)
        else:
            if self.D < 2 ** 11:
                return self.wht_slow.apply(x)
            else:
                return fwht_cpp.apply(x)

    @property
    def g_sigma(self):
        """
        Square roots of the covariance matrix for g (i.e. standard deviations of univariate Normal distributions).
        These are parameterized by g_sigma = softplus(g_rho) or equivalently g_sigma = 1 + log(1 + g_rho) to ensure
        non-negative values.
        """
        return F.softplus(self.g_rho)

    @property
    def kl(self):
        """
        KL divergence from the variational posterior to the prior.
        The prior is a multivariate normal distribution with mean vector zero and diagonal covariance with all elements
        being equal to self.lambda_.
        """
        return kl_diag_normal(
            self.g_mu,
            self.g_sigma,
            torch.zeros(self.D, device=self.g_mu.device),
            torch.ones(self.D, device=self.g_mu.device) * self.lambda_
        )

    def sample(self):
        """
        Sample a matrix W according to W = S1 @ H @ diag(g_tilde) @ H @ S2 with g_tilde drawn from the variational
        posterior with mean g_mu and standard deviation g_sigma.

        :return torch.Tensor: sampled matrix W.
        """
        epsilon = torch.randn(self.D, device=self.g_mu.device)
        g_tilde = self.g_mu + self.g_sigma * epsilon
        W = matmul_diag_left(self.s1, self.fwht(matmul_diag_left(g_tilde, self.fwht(torch.diag(self.s2)))))
        return W

    def forward(self, x):
        return F.linear(x, self.sample(), self.bias)


class WHVIStackedMatrix(nn.Module):
    def __init__(self, n_in, n_out, lambda_=1e-5, bias=False):
        """
        WHVI matrix with arbitrary dimensions (i.e. possibly non-square).
        A typical WHVI matrix is square with dimensions D x D where D == 2 ** d for some non-negative integer d.
        This class permits the use of arbitrarily-sized matrices by stacking appropriately-sized square matrices.

        :param n_in: number of input features.
        :param n_out: number of output features.
        :param lambda_: prior variance.
        :param boolean bias: if True, include a bias in the linear operation.
        """
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.lambda_ = lambda_

        self.D_in, self.D_out, self.padding, self.stack = self.setup_dimensions(n_in, n_out)
        self.weight_matrices = nn.ModuleList([
            WHVISquarePow2Matrix(self.D_in, lambda_=lambda_) for _ in range(self.stack)
        ])
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
        """
        KL divergence for this module.

        :return torch.Tensor, scalar: KL divergence from the variational posterior to the prior.
        """
        return sum(weight.kl for weight in self.weight_matrices)

    def sample(self):
        """
        Sample a weight matrix W by concatenating samples from matrix submodules.

        :return torch.Tensor: sampled weight matrix.
        """
        return torch.cat([weight.sample() for weight in self.weight_matrices])

    def forward(self, x):
        """
        Perform the forward pass.
        We pad the input x by zeros on the "right side" so that the length of the second dimension is self.D_in, then
        multiply this vector by a sampled weight matrix.
        We remove the elements, corresponding to the padded zeros after matrix multiplication. The length of the second
        dimension of the output thus becomes self.n_out.

        TODO pre-allocate x_padded (vector of zeros) if possible.

        :param torch.Tensor x: inputs of size (batch_size, self.n_in).
        :return torch.Tensor: outputs of size (batch_size, self.n_out).
        """
        x_padded = torch.zeros((*x.size()[:-1], self.D_in), device=x.device)  # Add the extra zeros
        x_padded[..., :self.n_in] = x
        output = F.linear(x_padded, self.sample(), self.bias)
        output = output[..., :self.n_out]  # Remove the extra elements
        return output


class WHVIColumnMatrix(nn.Module):
    def __init__(self, n_out, lambda_=1e-5, bias=False, transposed=False):
        """
        WHVI column matrix.
        Expects a single input feature and produces n_out output features.

        :param int n_out: number of output features.
        :param float lambda_: prior variance.
        :param boolean bias: if True, include a bias in the linear operation.
        :param boolean transposed: if True, treat the matrix as transposed. This effectively turns it into a layer that
                                   accepts multiple input features, but produces a single output feature.
        """
        super().__init__()
        self.D = n_out
        self.D_adjusted = 2 ** math.ceil(math.log(n_out, 2))
        self.weight_submodule = WHVISquarePow2Matrix(self.D_adjusted, lambda_=lambda_)
        self.transposed = transposed
        self.bias = nn.Parameter(torch.zeros(1, 1 if transposed else n_out)) if bias else None

    @property
    def kl(self):
        """
        KL divergence for this module.

        :return torch.Tensor, scalar: KL divergence from the variational posterior to the prior.
        """
        return self.weight_submodule.kl

    def sample(self):
        """
        Sample a weight matrix W by reshaping a sample from the weight submodule and taking its first self.D elements.

        :return torch.Tensor: sampled weight matrix.
        """
        matrix = torch.reshape(self.weight_submodule.sample(), (-1, 1))[:self.D]
        if self.transposed:
            return matrix.T
        return matrix

    def forward(self, x):
        return F.linear(x, self.sample(), self.bias)
