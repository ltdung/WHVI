import torch.nn as nn

from utils import is_pow_of_2
from weights import WHVIMatrix, WHVIStackedMatrix


class WHVI:
    def __init__(self):
        """
        Abstract class for WHVI layers.
        """
        self.kl = 0.0


class WHVILinear(nn.Module, WHVI):
    def __init__(self, n_in, n_out, lambda_=0.00001):
        """
        WHVI feed forward layer.

        :param n_in: input dimensionality.
        :param n_out: output dimensionality.
        :param lambda_: prior variance.
        """
        super().__init__()

        if n_in == n_out and is_pow_of_2(n_in):
            self.weight_submodule = WHVIMatrix(n_in, lambda_)
        else:
            self.weight_submodule = WHVIStackedMatrix(n_in, n_out, lambda_)

    @property
    def kl(self):
        return self.weight_submodule.kl

    def forward(self, x):
        return self.weight_submodule.forward(x)
