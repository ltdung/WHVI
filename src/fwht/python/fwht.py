import torch


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
