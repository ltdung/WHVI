from typing import Iterable, Tuple
import math
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

        :param Iterable[nn.Module] modules: iterable of torch modules to be passed to an underlying nn.Sequential
                                            object.
        :param loss_function: torch loss function to be used in training. This will be replaced in the future by an
                              explicit likelihood.
        :param int train_samples: number of Monte Carlo samples to draw during training.
        :param int eval_samples: number of Monte Carlo samples to draw during evaluation.
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
    log_rsqrt_2pi = math.log(1 / math.sqrt(2 * math.pi))

    def __init__(self, modules: Iterable[nn.Module], sigma=1.0, **kwargs):
        """
        WHVI neural network for regression.
        Assumes a normal likelihood for the data.
        Currently tested only for scalar regression targets.

        TODO consider good initializations for sigma.

        :param modules: iterable of nn.Module objects to be passed to an underlying nn.Sequential object.
        :param sigma: initial error tolerance.
        """
        super().__init__(modules, **kwargs)
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.current_kl = 0.0
        self.current_mnll = 0.0

    def mnll(self, y: torch.Tensor, y_hat: torch.Tensor, sigma: torch.Tensor):
        """
        Compute the mean negative log likelihood.
        The likelihood is assumed to be independent normal, i.e. y_hat[i] ~ N(y[i], sigma).

        TODO check that the likelihood is correctly computed.
        Argument y should have shape (n_observations, n_outputs), where n_observations represents the number of objects
        that have been processed in this pass and n_outputs the output dimensionality of the network.
        Argument y_hat should have shape (n_observations, n_outputs, n_mc_samples), where n_mc_samples represents the
        number of Monte Carlo samples that have been drawn to estimate this mean.

        :param torch.Tensor y: true target values.
        :param torch.Tensor y_hat: predicted target values.
        :param torch.Tensor sigma: sigma parameter for the normal likelihood.
        :return:
        """
        sums_of_squares = torch.zeros((y_hat.size()[2],), device=sigma.device)
        for j in range(len(sums_of_squares)):
            sums_of_squares[j] = torch.sum(torch.square(y - y_hat[..., j]))
        retval = -torch.mean(len(y) * (self.log_rsqrt_2pi - torch.log(sigma)) - 1 / (2 * sigma ** 2) * sums_of_squares)
        return retval

    def loss(self, x: torch.Tensor, y: torch.Tensor, ignore_kl=False, *args, **kwargs) -> torch.Tensor:
        """
        Compute ELBO given inputs x and targets y.
        The likelihood is assumed to be independent normal with variance sigma.

        :param torch.Tensor x: network input of shape (batch_size, in_dim).
        :param torch.Tensor y: network target of shape (batch_size, out_dim).
        :param boolean ignore_kl: if True, calculate loss as just MNLL.
        :return torch.Tensor: ELBO value.
        """
        self.current_mnll = self.mnll(y, self(x), self.sigma)
        self.current_kl = sum([layer.kl for layer in self.modules() if isinstance(layer, WHVI)])
        return self.current_mnll + self.current_kl if not ignore_kl else self.current_mnll

    def train_model(self, data_loader, optimizer, epochs1: int = 500, epochs2: int = 50000, pbar_update_period=20,
                    ignore_kl=False):
        """
        Train the model according to the procedure, described in the original paper.

        :param torch.utils.data.DataLoader data_loader: torch DataLoader object with training data.
        :param optimizer: torch optimizer.
        :param int epochs1: number of epochs to train for with fixed variance.
        :param int epochs2: number of epochs to train for with optimized variance.
        :param int pbar_update_period: the number of epochs as a period when the tqdm progress bar should be updated.
        :param boolean ignore_kl: if True, compute loss with only the MNLL term. Otherwise, include the KL term as well.
        """
        self.train()
        self.sigma.requires_grad = False  # Do not optimize sigma
        progress_bar = tqdm(
            range(epochs1),
            desc=f'[Fix. var.] WHVI (fix sigma), KL = {self.current_kl:.2f}, MNLL = {self.current_mnll:.2f}'
        )
        for epoch in progress_bar:
            for batch_index, (data_x, data_y) in enumerate(data_loader):
                loss = self.loss(data_x, data_y, ignore_kl=ignore_kl)
                loss.backward()
                optimizer.step()
                self.zero_grad(set_to_none=True)
            if epoch % pbar_update_period == 0:
                progress_bar.set_description(f'[Fix. var.] KL = {self.current_kl:.2f}, MNLL = {self.current_mnll:.2f}')

        self.sigma.requires_grad = True  # Optimize sigma
        progress_bar = tqdm(
            range(epochs2),
            desc=f'[Opt. var.] WHVI (fix sigma), KL = {self.current_kl:.2f}, MNLL = {self.current_mnll:.2f}'
        )
        for epoch in progress_bar:
            for batch_index, (data_x, data_y) in enumerate(data_loader):
                loss = self.loss(data_x, data_y, ignore_kl=ignore_kl)
                loss.backward()
                optimizer.step()
                self.zero_grad()
            if epoch % pbar_update_period == 0:
                progress_bar.set_description(f'[Opt. var.] KL = {self.current_kl:.2f}, MNLL = {self.current_mnll:.2f}')
        self.eval()

    def eval_model(self, X_test: torch.Tensor, y_test: torch.Tensor) -> Tuple[float, float]:
        """
        Evaluate the model on test data. Compute test error (MSE loss) and MNLL.

        :param torch.Tensor X_test: test data.
        :param torch.Tensor y_test: test targets.
        :return tuple: MSE loss and MNLL.
        """
        self.eval()
        y_pred = self(X_test)
        # TODO missing parameter sigma, but we use self.sigma anyway. Resolve (remove argument from declaration).
        test_mnll = self.mnll(y_test, y_pred)
        # kl = sum([layer.kl for layer in self.modules() if isinstance(layer, WHVI)])  # Unused
        test_error = F.mse_loss(y_pred.mean(dim=2), y_test)
        return float(test_error), float(test_mnll)
