import torch.nn as nn

from utils import is_pow_of_2
from weights import WHVISquarePow2Matrix, WHVIStackedMatrix, WHVIColumnMatrix


class WHVI:
    def __init__(self):
        """
        Abstract class for WHVI layers.
        """
        self.kl = 0.0


class WHVILinear(nn.Module, WHVI):
    def __init__(self, n_in, n_out, device, lambda_=0.00001, bias=False):
        """
        WHVI feed forward layer.

        :param n_in: input dimensionality.
        :param n_out: output dimensionality.
        :param lambda_: prior variance.
        """
        super().__init__()
        if n_in == 1:
            self.weight_submodule = WHVIColumnMatrix(n_out, device=device, lambda_=lambda_, bias=bias)
        elif n_out == 1:
            self.weight_submodule = WHVIColumnMatrix(n_in, device=device, lambda_=lambda_, transposed=True, bias=bias)
        elif n_in == n_out and is_pow_of_2(n_in):
            self.weight_submodule = WHVISquarePow2Matrix(n_in, device=device, lambda_=lambda_, bias=bias)
        else:
            self.weight_submodule = WHVIStackedMatrix(n_in, n_out, device=device, lambda_=lambda_, bias=bias)

    @property
    def kl(self):
        return self.weight_submodule.kl

    def forward(self, x):
        return self.weight_submodule.forward(x)
