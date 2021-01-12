from benchmark.classes import Benchmark, st_time
import torch
from src.walsh import build_H, fwht, FWHT


class WalshBenchmark(Benchmark):
    @st_time
    def matmul(self, H, A):
        H @ A

    @st_time
    def fwht_(self, A):
        fwht(A)

    @st_time
    def FWHT_(self, A):
        FWHT.apply(A)

    def benchmark_speed_small(self):
        D = 2 ** 2
        A = torch.randn(D, D)
        H = build_H(D)
        self.matmul(H, A)
        self.fwht_(A)
        self.FWHT_(A)

    def benchmark_speed_med(self):
        D = 2 ** 7
        A = torch.randn(D, D)
        H = build_H(D)
        self.matmul(H, A)
        self.fwht_(A)
        self.FWHT_(A)

    def benchmark_speed_large(self):
        D = 2 ** 12
        A = torch.randn(D, D)
        H = build_H(D)
        self.matmul(H, A)
        self.fwht_(A)
        self.FWHT_(A)


if __name__ == '__main__':
    print('')
    wb = WalshBenchmark()
    wb.main()
