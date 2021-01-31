import math
from torch.autograd import Function
import torch
import torch.nn as nn


class FWHTFunction(Function):
    @staticmethod
    def transform(u):
        n = u.shape[-1]
        m = int(math.log2(n))
        x = u.unsqueeze(-1)
        for _ in range(m)[::-1]:
            x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
        x = x.squeeze(-2)
        return x

    @staticmethod
    def forward(ctx, x):
        return FWHTFunction.transform(x)

    @staticmethod
    def backward(ctx, grad_output):
        return FWHTFunction.transform(grad_output)


class FWHT(nn.Module):
    def __init__(self):
        super(FWHT, self).__init__()

    def forward(self, x):
        return FWHTFunction.apply(x)
