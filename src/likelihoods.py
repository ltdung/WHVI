import torch
import torch.nn as nn


class Likelihood:
    def __init__(self):
        pass

    def nll(self, *args):
        return 0.0


class GaussianLikelihood(nn.Module, Likelihood):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma))

    def nll(self, y: torch.Tensor, y_hat: torch.Tensor):
        return -torch.distributions.Normal(y_hat, self.sigma).log_prob(y).sum()
