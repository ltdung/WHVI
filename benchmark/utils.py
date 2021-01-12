from benchmark.classes import Benchmark, st_time
from src.utils import matmul_diag_left
import torch


class UtilsBenchmark(Benchmark):
    @st_time
    def matmul(self, D, A):
        torch.diag(D) @ A

    @st_time
    def matmul_diag_left_(self, D, A):
        matmul_diag_left(D, A)

    def benchmark_speed_small(self):
        D = 2 ** 2
        A = torch.randn(D, D)
        diag_elements = torch.randn(D)
        self.matmul(diag_elements, A)
        self.matmul_diag_left_(diag_elements, A)

    def benchmark_speed_med(self):
        D = 2 ** 7
        A = torch.randn(D, D)
        diag_elements = torch.randn(D)
        self.matmul(diag_elements, A)
        self.matmul_diag_left_(diag_elements, A)

    def benchmark_speed_large(self):
        D = 2 ** 12
        A = torch.randn(D, D)
        diag_elements = torch.randn(D)
        self.matmul(diag_elements, A)
        self.matmul_diag_left_(diag_elements, A)


if __name__ == '__main__':
    UtilsBenchmark().main()
