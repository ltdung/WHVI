import unittest
import torch
from src.utils import build_H

import src.fwht.cpp.fwht as cpp_fwht
import src.fwht.cuda.fwht as cuda_fwht
import src.fwht.python.fwht_baseline as python_fwht


class WalshHadamard(unittest.TestCase):
    def test_single_column(self):
        a = torch.tensor([[1.0], [2.0], [3.0], [4.0]]).unsqueeze(0)
        reference = torch.tensor([[10.0], [-2.0], [-4.0], [0.0]]).unsqueeze(0)
        a = cpp_fwht.FWHTFunction.apply(a)
        self.assertTrue(torch.allclose(a, reference, atol=1e-5))

        a = torch.tensor([[0.0], [1.0], [2.0], [3.0]]).unsqueeze(0)
        reference = torch.tensor([[6.0], [-2.0], [-4.0], [0.0]]).unsqueeze(0)
        a = cpp_fwht.FWHTFunction.apply(a)
        self.assertTrue(torch.allclose(a, reference, atol=1e-5))

    def test_single_column_random(self):
        D = 2 ** 5
        H = build_H(D, torch.device("cpu"))
        for _ in range(30):
            a = torch.randn(D, 1)
            reference = (H @ a).unsqueeze(0)
            a = cpp_fwht.FWHTFunction.apply(a.unsqueeze(0))
            self.assertTrue(torch.allclose(a, reference, atol=1e-5))

    def test_matrix_random(self):
        D = 2 ** 5
        H = build_H(D, torch.device("cpu"))
        for _ in range(30):
            A = torch.randn(D, D)
            reference = (H @ A).unsqueeze(0)
            A = cpp_fwht.FWHTFunction.apply(A.unsqueeze(0))
            self.assertTrue(torch.allclose(A, reference, atol=1e-5))

    def test_python_cpp_equal(self):
        D = 2 ** 5
        for _ in range(30):
            A = torch.randn(D, D, device=torch.device("cpu"))
            B = torch.clone(A)
            A = cpp_fwht.FWHTFunction.apply(A.unsqueeze(0))
            B = python_fwht.FWHTFunction.apply(B.unsqueeze(0))
            self.assertTrue(torch.allclose(A, B, atol=1e-5))

    def test_cuda_simple(self):
        device = torch.device('cuda')
        D = 4
        num_batches = 2
        H = build_H(D, device)
        A = torch.randn(num_batches, D, device=device)
        reference = (H @ A.T).T
        output = cuda_fwht.FWHTFunction.apply(A)
        self.assertTrue(torch.allclose(reference, output))

    def test_cuda_large(self):
        device = torch.device('cuda')
        num_batches = 19
        D = 1024
        H = build_H(D, device)
        A = torch.randn(num_batches, D, device=device)
        reference = (H @ A.T).T
        output = cuda_fwht.FWHTFunction.apply(A)
        self.assertTrue(torch.allclose(output, reference, atol=1e-4))


if __name__ == '__main__':
    torch.manual_seed(0)
    unittest.main()
