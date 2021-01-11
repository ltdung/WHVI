import unittest
import torch
from src.walsh import fwht, build_H


class WalshHadamard(unittest.TestCase):
    def test_single_column(self):
        a = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        reference = torch.tensor([[10.0], [-2.0], [-4.0], [0.0]])
        fwht(a)
        self.assertTrue(torch.allclose(a, reference, atol=1e-5))

        a = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        reference = torch.tensor([[6.0], [-2.0], [-4.0], [0.0]])
        fwht(a)
        self.assertTrue(torch.allclose(a, reference, atol=1e-5))

    def test_single_column_automated(self):
        D = 2 ** 5
        H = build_H(D)
        for _ in range(30):
            a = torch.randn(D, 1)
            reference = H @ a
            fwht(a)
            self.assertTrue(torch.allclose(a, reference, atol=1e-5))

    def test_matrix_automated(self):
        D = 2 ** 5
        H = build_H(D)
        for _ in range(30):
            A = torch.randn(D, D)
            reference = H @ A
            fwht(A)
            self.assertTrue(torch.allclose(A, reference, atol=1e-5))


if __name__ == '__main__':
    torch.manual_seed(0)
    unittest.main()
