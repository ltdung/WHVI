import math
import torch
import torch.nn as nn
from torch.autograd import Function

from src.utils import build_H, is_pow_of_2


class WHT_matmul:
    def __init__(self):
        """
        Class for the batched Walsh-Hadamard transform using matrix multiplication.
        """
        self.H = None
        self.H_built = False

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies WHT to input tensor x.

        :param torch.Tensor x: inputs of shape (batch_size, D)
        :return: transformed inputs, multiplied by H.
        """

        D = x.size()[1]
        assert is_pow_of_2(D)
        assert x.dim() == 2

        if not self.H_built:
            self.H = build_H(D, x.device)
            self.H_built = True
        return (self.H @ x.T).T


class FWHTFunction(Function):
    """
    Torch autograd function for vectorized batched FWHT.
    """

    @staticmethod
    def transform(x: torch.Tensor):
        """
        Vectorized batched fast Walsh-Hadamard transform (FWHT).

        :param torch.Tensor x: tensor of inputs
        :return: transformed inputs according to FWHT.
        """
        D = x.size()[1]
        assert is_pow_of_2(D)
        assert x.dim() == 2

        x_out = x.unsqueeze(2)
        for _ in range(int(math.log2(x.shape[1])))[::-1]:
            x_out = torch.cat((x_out[:, ::2] + x_out[:, 1::2], x_out[:, ::2] - x_out[:, 1::2]), dim=2)
        return x_out.squeeze(1)

    @staticmethod
    def forward(ctx, x):
        return FWHTFunction.transform(x)

    @staticmethod
    def backward(ctx, grad_output):
        return FWHTFunction.transform(grad_output)


class FWHT(nn.Module):
    def __init__(self):
        """
        Module for FWHT, currently unused.
        """
        super(FWHT, self).__init__()

    def forward(self, x):
        return FWHTFunction.apply(x)
