import torch
import torch.nn as nn


class Likelihood:
    def __init__(self):
        pass

    def nll(self, *args):
        return 0.0

    def mnll(self, *args):
        return 0.0

    def mnll_batch_estimate(self, *args):
        return 0.0


class GaussianLikelihood(nn.Module, Likelihood):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma))

    def nll(self, y: torch.Tensor, y_hat: torch.Tensor):
        """

        :param torch.Tensor, scalar y:
        :param torch.Tensor, scalar y_hat:
        :return torch.Tensor, scalar:
        """
        return -torch.distributions.Normal(y_hat, self.sigma).log_prob(y)

    def mnll(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """

        :param torch.Tensor, scalar y:
        :param torch.Tensor, vector y_hat:
        :return torch.Tensor, scalar:
        """
        return (-torch.distributions.Normal(y_hat, self.sigma).log_prob(y)).mean()

    def mnll_batch_estimate(self, y: torch.Tensor, y_hat: torch.Tensor, n: int) -> torch.Tensor:
        """

        :param torch.Tensor y: true target values.
        :param torch.Tensor y_hat: predicted target values.
        :param int n: data set size (training set size when training, test set size when testing).
        :return torch.Tensor, scalar: mean negative log likelihood.
        """
        mnll = torch.tensor([0.0], device=y.device)
        m = y.size()[0]  # Batch size
        for j in range(m):
            tmp = self.mnll(y[j], y_hat[j])
            mnll += tmp
        mnll *= n / m
        return mnll[0]
