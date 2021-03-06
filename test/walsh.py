import unittest
import torch
from src.utils import build_H

import src.fwht.cpp.fwht as cpp_fwht
import src.fwht.cuda.fwht as cuda_fwht
import src.fwht.python.fwht as python_fwht


class WalshHadamard(unittest.TestCase):
    def test_single_column(self):
        a = torch.tensor([[1.0], [2.0], [3.0], [4.0]]).T
        reference = torch.tensor([[10.0], [-2.0], [-4.0], [0.0]]).T
        a = cpp_fwht.FWHTFunction.apply(a)
        self.assertTrue(torch.allclose(a, reference, atol=1e-5))

        a = torch.tensor([[0.0], [1.0], [2.0], [3.0]]).T
        reference = torch.tensor([[6.0], [-2.0], [-4.0], [0.0]]).T
        a = cpp_fwht.FWHTFunction.apply(a)
        self.assertTrue(torch.allclose(a, reference, atol=1e-5))

    def test_single_column_random(self):
        D = 2 ** 5
        H = build_H(D, torch.device("cpu"))
        for _ in range(30):
            a = torch.randn(1, D)
            reference = (H @ a.T).T
            a = cpp_fwht.FWHTFunction.apply(a)
            self.assertTrue(torch.allclose(a, reference, atol=1e-5))

    def test_slow_WHT(self):
        device = torch.device('cpu')
        D = 2 ** 5
        batch_size = 40
        A = torch.randn(batch_size, D, device=device)
        ret1 = python_fwht.WHT_matmul().apply(A)
        H = build_H(D, device)
        ret2 = (H @ A.T).T
        self.assertTrue(torch.allclose(ret1, ret2))

    def test_matrix_random(self):
        D = 2 ** 5
        batch_size = 40
        H = build_H(D, torch.device("cpu"))
        for _ in range(30):
            A = torch.randn(batch_size, D)
            reference = (H @ A.T).T
            A = cpp_fwht.FWHTFunction.apply(A)
            self.assertTrue(torch.allclose(A, reference, atol=1e-5))

    def test_python_cpp_equal(self):
        D = 2 ** 5
        batch_size = 40
        for _ in range(30):
            A = torch.randn(batch_size, D, device=torch.device("cpu"))
            B = torch.clone(A)
            A = cpp_fwht.FWHTFunction.apply(A)
            B = python_fwht.FWHTFunction.apply(B)
            self.assertTrue(torch.allclose(A, B, atol=1e-3))

    def test_cuda_simple(self):
        device = torch.device('cuda')
        D = 4
        batch_size = 2
        H = build_H(D, device)
        A = torch.randn(batch_size, D, device=device)
        reference = (H @ A.T).T
        output = cuda_fwht.FWHTFunction.apply(A)
        self.assertTrue(torch.allclose(reference, output))

    def test_cuda_large(self):
        device = torch.device('cuda')
        batch_size = 19
        D = 1024
        H = build_H(D, device)
        A = torch.randn(batch_size, D, device=device)
        reference = (H @ A.T).T
        output = cuda_fwht.FWHTFunction.apply(A)
        self.assertTrue(torch.allclose(output, reference, atol=1e-4))


if __name__ == '__main__':
    torch.manual_seed(0)
    unittest.main()
