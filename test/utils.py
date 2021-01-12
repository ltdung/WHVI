import unittest
import torch

from src.utils import matmul_diag_left, matmul_diag_right


class Matmul(unittest.TestCase):
    def test_matmul_diag_left(self):
        D = 2 ** 5
        for _ in range(30):
            A = torch.randn(D, D)
            D_elements = torch.randn(D)
            self.assertTrue(torch.allclose(torch.diag(D_elements) @ A, matmul_diag_left(D_elements, A)))

    def test_matmul_diag_right(self):
        D = 2 ** 5
        for _ in range(30):
            A = torch.randn(D, D)
            D_elements = torch.randn(D)
            self.assertTrue(torch.allclose(A @ torch.diag(D_elements), matmul_diag_right(A, D_elements)))


if __name__ == '__main__':
    unittest.main()
