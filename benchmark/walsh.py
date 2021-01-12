# TODO compare the performance of the implemented FWHT to regular matrix multiplication.
from benchmark.classes import Benchmark, st_time
import torch
from src.walsh import build_H, fwht


class WalshBenchmark(Benchmark):
    @st_time
    def matmul(self, H, A):
        ret = H @ A

    @st_time
    def fwht_(self, A):
        fwht(A)

    def benchmark_speed_small(self):
        D = 2 ** 2
        A = torch.randn(D, D)
        H = build_H(D)
        self.matmul(H, A)
        self.fwht_(A)

    def benchmark_speed_med(self):
        D = 2 ** 7
        A = torch.randn(D, D)
        H = build_H(D)
        self.matmul(H, A)
        self.fwht_(A)

    def benchmark_speed_large(self):
        D = 2 ** 12
        A = torch.randn(D, D)
        H = build_H(D)
        self.matmul(H, A)
        self.fwht_(A)


if __name__ == '__main__':
    print('')
    wb = WalshBenchmark()
    wb.main()
