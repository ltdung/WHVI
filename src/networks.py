from typing import Iterable
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import WHVILinear, WHVI


class WHVINetwork(nn.Sequential):
    def __init__(self, modules: Iterable[nn.Module], loss_function=F.mse_loss, train_samples=1, eval_samples=64):
        """
        Sequential neural network which supports WHVI layers.
        The output should be a vector.

        :param args: valid sequence of network layers of type torch.nn.Module.
        :param int train_samples: number of Monte Carlo samples to draw during training.
        :param int eval_samples: number of Monte Carlo samples to draw during evaluation.


        Example:
        >>> modules = [nn.Linear(28 * 28, 128), nn.ReLU(), WHVILinear(128), nn.ReLU(), nn.Linear(128, 10)]
        >>> net = WHVINetwork(modules)
        >>> net(torch.randn(100, 64))
        """
        super().__init__(*modules)
        self.train_samples = train_samples
        self.eval_samples = eval_samples
        self.loss_function = loss_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagates input x through the network.

        :param torch.Tensor x: network input of shape (batch_size, in_dim).
        :return torch.Tensor: network output of shape (batch_size, out_dim, n_samples).
        """
        assert x.dim() == 2, "Input shape must be (batch_size, in_dim)"
        batch_size = x.size()[0]
        n_samples = self.train_samples if self.training else self.eval_samples

        predictions = []
        for _ in range(n_samples):
            batch_predictions = super(WHVINetwork, self).forward(x)
            predictions.append(torch.reshape(batch_predictions, (batch_size, batch_predictions.size()[-1], 1)))
        predictions = torch.cat(predictions, dim=2)

        assert predictions.dim() == 3
        return predictions

    def loss(self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Compute ELBO given inputs x and target y.

        :param torch.Tensor x: network input of shape (batch_size, in_dim).
        :param torch.Tensor y: network target of shape (batch_size, out_dim).
        :return torch.Tensor: ELBO value.
        """
        n_samples = self.train_samples if self.training else self.eval_samples
        kl = sum([layer.kl for layer in self.modules() if isinstance(layer, WHVI)])
        nll_samples = torch.zeros(n_samples)
        predictions = self(x)  # Obtain n_samples of predictions
        for i in range(n_samples):
            nll_samples[i] = self.loss_function(predictions[..., i], y)
        mean_nll = nll_samples.mean()
        return mean_nll + kl


class WHVIRegression(WHVINetwork):
    neg_half_log_2pi = -0.5 * np.log(2 * np.pi)

    def __init__(self, modules: Iterable[nn.Module], sigma=1.0, **kwargs):
        super().__init__(modules, **kwargs)
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.current_kl = 0.0
        self.current_mnll = 0.0

    def likelihood(self, y: torch.Tensor, y_hat: torch.Tensor, sigma: torch.Tensor):
        n = len(y)
        n_samples = y_hat.size()[2]
        squared_terms = torch.zeros((n,))
        for j in range(n_samples):
            squared_terms[j] = torch.sum(torch.square(y - y_hat[..., j])) / sigma ** 2
        return n * (self.neg_half_log_2pi - torch.log(sigma)) - 0.5 * torch.mean(squared_terms)

    def loss(self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Compute ELBO given inputs x and target y.
        The likelihood is assumed to be according to the argument sigma.

        :param torch.Tensor x: network input of shape (batch_size, in_dim).
        :param torch.Tensor y: network target of shape (batch_size, out_dim).
        :return torch.Tensor: ELBO value.
        """
        self.current_mnll = -self.likelihood(y, self(x), self.sigma)
        self.current_kl = sum([layer.kl for layer in self.modules() if isinstance(layer, WHVI)])
        return self.current_mnll + self.current_kl

    def train_model(self, data_loader, optimizer, epochs1: int = 500, epochs2: int = 50000, pbar_update_period=20):
        self.train()
        self.sigma.requires_grad = False  # Do not optimize sigma
        progress_bar = tqdm(
            range(epochs1),
            desc=f'[Fix. var.] WHVI (fix sigma), KL = {self.current_kl:.2f}, MNLL = {self.current_mnll:.2f}'
        )
        for epoch in progress_bar:
            for batch_index, (data_x, data_y) in enumerate(data_loader):
                loss = self.loss(data_x, data_y, self.sigma)
                loss.backward()
                optimizer.step()
                self.zero_grad()
            if epoch % pbar_update_period == 0:
                progress_bar.set_description(f'[Fix. var.] KL = {self.current_kl:.2f}, MNLL = {self.current_mnll:.2f}')

        self.sigma.requires_grad = True  # Optimize sigma
        progress_bar = tqdm(
            range(epochs2),
            desc=f'[Opt. var.] WHVI (fix sigma), KL = {self.current_kl:.2f}, MNLL = {self.current_mnll:.2f}'
        )
        for epoch in progress_bar:
            for batch_index, (data_x, data_y) in enumerate(data_loader):
                loss = self.loss(data_x, data_y, self.sigma)
                loss.backward()
                optimizer.step()
                self.zero_grad()
            if epoch % pbar_update_period == 0:
                progress_bar.set_description(f'[Opt. var.] KL = {self.current_kl:.2f}, MNLL = {self.current_mnll:.2f}')

        self.eval()
