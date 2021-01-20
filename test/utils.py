import unittest
import torch

from src.utils import matmul_diag_left, matmul_diag_right, kl_diag_normal


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

    def test_kl(self):
        for _ in range(30):
            mu1 = torch.randn(10)
            sd1 = torch.exp(torch.randn(10))
            mu2 = torch.randn(10)
            sd2 = torch.exp(torch.randn(10))

            torch_result = torch.distributions.kl.kl_divergence(
                torch.distributions.MultivariateNormal(mu1, torch.diag(sd1)),
                torch.distributions.MultivariateNormal(mu2, torch.diag(sd2))
            )
            own_result = kl_diag_normal(mu1, sd1, mu2, sd2)
            self.assertTrue(torch.allclose(torch_result, own_result))


if __name__ == '__main__':
    unittest.main()
