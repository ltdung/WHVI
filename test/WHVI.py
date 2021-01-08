import unittest
from src.layers import BasicWHVILinear
import torch

torch.manual_seed(0)


class TestWHVI(unittest.TestCase):
    def test_output_size(self):
        for M in [3, 6, 10, 17, 50, 123]:
            for D in [2, 4, 8, 16, 32]:
                layer = BasicWHVILinear(D)
                self.assertEqual(layer.sample_W().size(), (D, D))

                x = torch.zeros(M, D)
                out = layer(x)
                self.assertEqual(out.size(), (M, D))


if __name__ == '__main__':
    unittest.main()
