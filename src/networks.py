import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Iterable, Tuple
from tqdm import tqdm
from src.layers import WHVI
from src.likelihoods import GaussianLikelihood, Likelihood
import pathlib


class WHVINetwork(nn.Module, WHVI):
    def __init__(self, modules: Iterable[nn.Module], likelihood: Likelihood, train_samples=1, eval_samples=64):
        """
        Sequential neural network which supports WHVI layers.
        The output should be a vector.

        :param Iterable[nn.Module] modules: iterable of torch modules to be passed to an underlying nn.Sequential
                                            object.
        :param likelihood: Likelihood object to be used in training.
        :param int train_samples: number of Monte Carlo samples to draw during training.
        :param int eval_samples: number of Monte Carlo samples to draw during evaluation.
        """
        super().__init__()
        self.sequential = nn.Sequential(*modules)
        self.likelihood = likelihood
        self.train_samples = train_samples
        self.eval_samples = eval_samples
        self.current_mnll = 0.0
        self.current_kl = 0.0

    @property
    def kl(self):
        return sum([m.kl for m in self.sequential.children() if 'kl' in dir(m)])

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
            batch_predictions = self.sequential.forward(x)
            predictions.append(torch.reshape(batch_predictions, (batch_size, batch_predictions.size()[-1], 1)))
        predictions = torch.cat(predictions, dim=2)

        assert predictions.dim() == 3
        return predictions

    def loss(self, x: torch.Tensor, y: torch.Tensor, n: int, ignore_kl=False) -> torch.Tensor:
        """
        Compute ELBO given inputs x and targets y.
        The likelihood is assumed to be independent normal with variance sigma.

        :param torch.Tensor x: network input of shape (batch_size, in_dim).
        :param torch.Tensor y: network target of shape (batch_size, out_dim).
        :param int n: data set size (training set size when training, test set size when testing).
        :param boolean ignore_kl: if True, calculate loss as just MNLL.
        :return torch.Tensor: ELBO value.
        """
        self.current_mnll = self.likelihood.mnll_batch_estimate(y, self(x), n)
        self.current_kl = self.kl
        return self.current_mnll + self.current_kl if not ignore_kl else self.current_mnll

    def train_model(self, data_loader, optimizer, scheduler, epochs1: int = 500, epochs2: int = 5000,
                    pbar_update_period=20, ignore_kl=False, checkpoint_dir=None):
        self.train()
        self.likelihood.requires_grad = False  # Do not optimize likelihood parameters
        pbar = tqdm(range(epochs1), desc=f'[Fixed LH] KL = {self.current_kl:.2f}, MNLL = {self.current_mnll:.2f}')
        for epoch in pbar:
            for batch_index, (data_x, data_y) in enumerate(data_loader):
                loss = self.loss(data_x, data_y, n=len(data_loader.dataset), ignore_kl=ignore_kl)
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.zero_grad(set_to_none=True)
            if epoch % pbar_update_period == 0:
                pbar.set_description(f'[Fixed LH] KL = {self.current_kl:.2f}, MNLL = {self.current_mnll:.2f}')

        self.likelihood.requires_grad = True  # Optimize likelihood parameters
        pbar = tqdm(range(epochs2), desc=f'[Optimized LH] KL = {self.current_kl:.2f}, MNLL = {self.current_mnll:.2f}')
        for epoch in pbar:
            for batch_index, (data_x, data_y) in enumerate(data_loader):
                loss = self.loss(data_x, data_y, n=len(data_loader.dataset), ignore_kl=ignore_kl)
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.zero_grad()
            if epoch % 5000 == 0 and checkpoint_dir is not None:
                torch.save(self.state_dict(), pathlib.Path(checkpoint_dir) / f'epoch-{epoch}.pth')  # Save model state
            if epoch % pbar_update_period == 0:
                pbar.set_description(f'[Optimized LH] KL = {self.current_kl:.2f}, MNLL = {self.current_mnll:.2f}')
        self.eval()

    def eval_model(self, X_test: torch.Tensor, y_test: torch.Tensor, loss) -> Tuple[float, float]:
        """
        Evaluate the model on test data. Compute test error (loss value) and MNLL.

        :param torch.Tensor X_test: test data.
        :param torch.Tensor y_test: test targets.
        :param loss: loss function which accepts predicted values of shape (batch_size, out_dim, mc_samples) as the
            first and target values of shape (batch_size, out_dim) as the second argument. Returns a scalar tensor.
        :return tuple: loss value and MNLL.
        """
        self.eval()
        y_pred = self(X_test)
        test_mnll = self.likelihood.mnll_batch_estimate(y_test, y_pred, n=y_test.size()[0])
        test_error = loss(y_pred, y_test)
        return float(test_error), float(test_mnll)


class WHVIRegression(WHVINetwork):
    def __init__(self, modules: Iterable[nn.Module], sigma: float = 1.0, **kwargs):
        """
        WHVI neural network for regression.
        Assumes a normal likelihood for the data.
        Currently tested only for scalar regression targets.

        :param modules: iterable of nn.Module objects to be passed to an underlying WHVINetwork object.
        :param sigma: initial error tolerance.
        """
        super().__init__(modules, likelihood=GaussianLikelihood(sigma), **kwargs)

    def eval_model(self, X_test: torch.Tensor, y_test: torch.Tensor,
                   loss=lambda y_pred, y_true: torch.sqrt(
                       F.mse_loss(y_pred.mean(dim=2).flatten(), y_true.flatten()))) -> Tuple[float, float]:
        return super().eval_model(X_test, y_test, loss)
