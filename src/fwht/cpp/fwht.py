from torch.autograd import Function
import torch.nn as nn

import fwht_cpp


class FWHTFunction(Function):
    """
    Python frontend for the batched FWHT C++ implementation.
    """

    @staticmethod
    def forward(ctx, x):
        return fwht_cpp.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        return fwht_cpp.backward(grad_output)


class FWHT(nn.Module):
    """
    Python frontend for the batched FWHT C++ implementation as a torch module.
    """

    def __init__(self):
        super(FWHT, self).__init__()

    def forward(self, x):
        return FWHTFunction.apply(x)
