import torch

from utils import build_H


class WHT_matmul:
    def __init__(self):
        self.H = None
        self.H_built = False

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if not self.H_built:
            D = x.size()[0]
            self.H = build_H(D, x.device)
            self.H_built = True
        return self.H @ x


class FWHT(torch.nn.Module):
    def __init__(self):
        super(FWHT, self).__init__()

    def forward(self, x):
        n = x.size(1)
        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    x[:, j, :], x[:, j + h, :] = x[:, j, :] + x[:, j + h, :], x[:, j, :] - x[:, j + h, :]
            h *= 2
        return x
