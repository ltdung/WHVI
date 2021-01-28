import torch
import torch.nn as nn


class Likelihood:
    def __init__(self):
        pass

    def mnll_batch_estimate(self, *args, **kwargs):
        return 0.0


class GaussianLikelihood(nn.Module, Likelihood):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma))

    def mnll_batch_estimate(self, y: torch.Tensor, y_hat: torch.Tensor, n: int) -> torch.Tensor:
        """

        :param torch.Tensor y: true target values.
        :param torch.Tensor y_hat: predicted target values.
        :param int n: data set size (training set size when training, test set size when testing).
        :return torch.Tensor, scalar: mean negative log likelihood.
        """
        m = y_hat.size()[0]
        n_mc = y_hat.size()[2]
        mnll = -n / (m * n_mc) * torch.distributions.Normal(y_hat.T.flatten(), self.sigma).log_prob(
            y.flatten().repeat(n_mc)).sum()
        return mnll
