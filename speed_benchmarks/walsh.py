from speed_benchmarks.classes import Benchmark, st_time
import torch
from src.walsh import build_H, fwht, FWHT, FWHT_diag


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

    @st_time
    def FWHT_diag_(self, D):
        FWHT_diag.apply(D)

    def benchmark_FWHT_speed_small(self):
        D = 2 ** 2
        A = torch.randn(D, D)
        H = build_H(D)
        self.matmul(H, A)
        self.fwht_(A)
        self.FWHT_(A)

    def benchmark_FWHT_speed_med(self):
        D = 2 ** 7
        A = torch.randn(D, D)
        H = build_H(D)
        self.matmul(H, A)
        self.fwht_(A)
        self.FWHT_(A)

    def benchmark_FWHT_speed_large(self):
        D = 2 ** 12
        A = torch.randn(D, D)
        H = build_H(D)
        self.matmul(H, A)
        self.fwht_(A)
        self.FWHT_(A)

    def benchmark_FWHT_speed_very_large(self):
        D = 2 ** 13
        A = torch.randn(D, D)
        H = build_H(D)
        self.matmul(H, A)
        self.fwht_(A)
        self.FWHT_(A)

    def benchmark_FWHT_diag_speed_small(self):
        D = 2 ** 2
        elements = torch.randn(D)
        H = build_H(D)
        self.matmul(H, torch.diag(elements))
        self.FWHT_diag_(elements)

    def benchmark_FWHT_diag_speed_med(self):
        D = 2 ** 7
        elements = torch.randn(D)
        H = build_H(D)
        self.matmul(H, torch.diag(elements))
        self.FWHT_diag_(elements)

    def benchmark_FWHT_diag_speed_large(self):
        D = 2 ** 12
        elements = torch.randn(D)
        H = build_H(D)
        self.matmul(H, torch.diag(elements))
        self.FWHT_diag_(elements)

    def benchmark_FWHT_diag_speed_very_large(self):
        D = 2 ** 13
        elements = torch.randn(D)
        H = build_H(D)
        self.matmul(H, torch.diag(elements))
        self.FWHT_diag_(elements)


if __name__ == '__main__':
    wb = WalshBenchmark()
    wb.main()
