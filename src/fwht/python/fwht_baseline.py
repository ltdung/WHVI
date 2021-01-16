from torch.autograd import Function
import torch
import torch.nn as nn


class FWHTFunction(Function):
    @staticmethod
    def transform(x):
        n = x.size(1)
        h = 1
        y = torch.clone(x)
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    y[:, j, :], y[:, j + h, :] = y[:, j, :] + y[:, j + h, :], y[:, j, :] - y[:, j + h, :]
            h *= 2
        return y

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
