import math
from torch.autograd import Function
import torch
import torch.nn as nn


class FWHTFunction(Function):
    @staticmethod
    def transform(x_in):
        x_out = x_in.unsqueeze(2)
        for _ in range(int(math.log2(x_in.shape[1])))[::-1]:
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
        super(FWHT, self).__init__()

    def forward(self, x):
        return FWHTFunction.apply(x)
