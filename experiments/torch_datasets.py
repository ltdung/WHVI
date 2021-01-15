from abc import ABC
import torch
import numpy as np

from torch.utils.data import Dataset


class ToyDataset(Dataset, ABC):
    def __init__(self, device, n=128):
        self.device = device
        self.n = n

        # It seems that y = f(x) + eps where eps ~ N(0, exp(-3)). The authors do not explicitly say what f is.
        # For this reason, we define a similar polynomial and see if we observe a similar pattern.
        # The polynomial coefficients are chosen based on observed extrema in the plot.
        self.coef = self.make_poly_coef()

        x = torch.distributions.Uniform(-1, 2).sample((n,))
        x = x[torch.where((x < 0.4) | (x > 1.4))]
        y = self.f(x) + torch.distributions.Normal(0, np.sqrt(np.exp(-3))).sample((len(x),))

        self.x = torch.reshape(x, (-1, 1)).to(device)
        self.y = torch.reshape(y, (-1, 1)).to(device)

    def make_poly_coef(self):
        xs_poly = [-2.0, -1.5, -0.8, 0.0, 0.5, 1.4, 2.0, 2.7, 3.0]
        ys_poly = [1.2, 1.5, 2.0, 0.5, -0.5, 1.2, 0.0, 1.0, 1.3]
        vmat = np.vander(xs_poly, len(xs_poly))
        coef = np.linalg.solve(vmat, ys_poly)
        return coef

    def f(self, x):
        return torch.from_numpy(np.polyval(self.coef, x))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
