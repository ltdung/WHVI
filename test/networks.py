import unittest

import torch
import torch.nn as nn
from src.networks import WHVILinear, WHVINetwork


class WHVINetworkTestCase(unittest.TestCase):
    def test_dimensions(self):
        for k in range(1, 21):
            net = WHVINetwork([
                nn.Linear(1, 8), WHVILinear(8), nn.Linear(8, k)
            ], train_samples=5, eval_samples=6)

            net.train()
            out = net(torch.randn(50, 1))
            self.assertEqual(out.size(), (50, k, 5))

            net.eval()
            out = net(torch.randn(50, 1))
            self.assertEqual(out.size(), (50, k, 6))


if __name__ == '__main__':
    unittest.main()
