from benchmarks.classes import Benchmark, st_time
import torch
from src.utils import build_H
from src.fwht.cpp.fwht import FWHTFunction


class WalshBenchmark(Benchmark):
    @st_time
    def matmul(self, H, A):
        H @ A

    @st_time
    def FWHTFunction_(self, A):
        FWHTFunction.apply(A)

    def benchmark_FWHT_speed_small(self):
        D = 2 ** 2
        A = torch.randn(D, D)
        H = build_H(D, device=torch.device("cpu"))
        self.matmul(H, A)
        self.FWHTFunction_(A.unsqueeze(0))

    def benchmark_FWHT_speed_med(self):
        D = 2 ** 7
        A = torch.randn(D, D)
        H = build_H(D, device=torch.device("cpu"))
        self.matmul(H, A)
        self.FWHTFunction_(A.unsqueeze(0))

    def benchmark_FWHT_speed_large(self):
        D = 2 ** 12
        A = torch.randn(D, D)
        H = build_H(D, device=torch.device("cpu"))
        self.matmul(H, A)
        self.FWHTFunction_(A.unsqueeze(0))

    def benchmark_FWHT_speed_very_large(self):
        D = 2 ** 13
        A = torch.randn(D, D)
        H = build_H(D, device=torch.device("cpu"))
        self.matmul(H, A)
        self.FWHTFunction_(A.unsqueeze(0))


if __name__ == '__main__':
    wb = WalshBenchmark()
    wb.main()
