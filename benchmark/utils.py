from benchmark.classes import Benchmark, st_time
from src.utils import matmul_diag_left, matmul_diag_right
import torch


class MatmulDiagBenchmark(Benchmark):
    @st_time
    def matmul_left(self, D, A):
        torch.diag(D) @ A

    @st_time
    def matmul_right(self, A, D):
        A @ torch.diag(D)

    @st_time
    def matmul_diag_left_(self, D, A):
        matmul_diag_left(D, A)

    @st_time
    def matmul_diag_right_(self, A, D):
        matmul_diag_right(A, D)

    def benchmark_matmul_diag_left_small(self):
        D = 2 ** 2
        A = torch.randn(D, D)
        diag_elements = torch.randn(D)
        self.matmul_left(diag_elements, A)
        self.matmul_diag_left_(diag_elements, A)

    def benchmark_matmul_diag_left_med(self):
        D = 2 ** 7
        A = torch.randn(D, D)
        diag_elements = torch.randn(D)
        self.matmul_left(diag_elements, A)
        self.matmul_diag_left_(diag_elements, A)

    def benchmark_matmul_diag_left_large(self):
        D = 2 ** 12
        A = torch.randn(D, D)
        diag_elements = torch.randn(D)
        self.matmul_left(diag_elements, A)
        self.matmul_diag_left_(diag_elements, A)

    def benchmark_matmul_diag_right_small(self):
        D = 2 ** 2
        A = torch.randn(D, D)
        diag_elements = torch.randn(D)
        self.matmul_right(A, diag_elements)
        self.matmul_diag_right_(A, diag_elements)

    def benchmark_matmul_diag_right_med(self):
        D = 2 ** 7
        A = torch.randn(D, D)
        diag_elements = torch.randn(D)
        self.matmul_right(A, diag_elements)
        self.matmul_diag_right_(A, diag_elements)

    def benchmark_matmul_diag_right_large(self):
        D = 2 ** 12
        A = torch.randn(D, D)
        diag_elements = torch.randn(D)
        self.matmul_right(A, diag_elements)
        self.matmul_diag_right_(A, diag_elements)


if __name__ == '__main__':
    MatmulDiagBenchmark().main()
