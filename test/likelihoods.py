import unittest
import torch
from src.likelihoods import GaussianLikelihood
import numpy as np


class LikelihoodCase(unittest.TestCase):
    def test_gaussian_vectorization(self):
        n = 12
        m = 4
        n_mc = 2
        sigma = 1.0

        y = torch.reshape(torch.tensor([0., 1., 2., -1.]), (-1, 1))
        y_hat = torch.tensor([[0.2, 1.1, 2.2, -1.3], [-0.1, 1.05, 2, -1.1]]).T.unsqueeze(1)

        # According to the formula
        target = 0
        for j in range(m):
            tmp = 0
            for i in range(n_mc):
                tmp += -(np.log(1 / (np.sqrt(2 * np.pi) * sigma)) - 1 / 2 * ((y[j] - y_hat[j, 0, i]) / sigma) ** 2)
            tmp /= n_mc
            target += tmp
        target *= n / m
        target = float(target)

        # Vectorized
        computed = float(GaussianLikelihood(sigma=sigma).mnll_batch_estimate(y, y_hat, n))

        self.assertAlmostEqual(target, computed, delta=1e-4)

    def test_gaussian_vectorization_random(self):
        n = 116
        m = 24
        n_mc = 80
        sigma = 15.21

        y = torch.randn((m, 1))
        y_hat = torch.randn((m, 1, n_mc))

        # According to the formula
        target = 0
        for j in range(m):
            tmp = 0
            for i in range(n_mc):
                tmp += -(np.log(1 / (np.sqrt(2 * np.pi) * sigma)) - 1 / 2 * ((y[j] - y_hat[j, 0, i]) / sigma) ** 2)
            tmp /= n_mc
            target += tmp
        target *= n / m
        target = float(target)

        # Vectorized
        computed = float(GaussianLikelihood(sigma=sigma).mnll_batch_estimate(y, y_hat, n))

        self.assertAlmostEqual(target, computed, delta=1e-4)


if __name__ == '__main__':
    unittest.main()
