import unittest

import torch
import torch.nn as nn
from src.networks import WHVILinear, WHVIRegression


class RegressionTestCase(unittest.TestCase):
    def test_dimensions(self):
        for k in range(1, 21):
            net = WHVIRegression([
                nn.Linear(1, 8), nn.ReLU(), WHVILinear(8), nn.ReLU(), nn.Linear(8, k)
            ], train_samples=5, eval_samples=6)

            net.train()
            out = net(torch.randn(50, 1))
            self.assertEqual(out.size(), (50, k, 5))

            net.eval()
            out = net(torch.randn(50, 1))
            self.assertEqual(out.size(), (50, k, 6))


if __name__ == '__main__':
    unittest.main()
