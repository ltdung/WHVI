import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import build_H

import src.fwht.cpp.fwht as cpp_fwht
import src.fwht.cuda.fwht as cuda_fwht

"""
Measure speed of matrix multiplication and fast Walsh-Hadamard transform (Python and C++ implementations) on the CPU.
Measure speed of matrix multiplication and fast Walsh-Hadamard transform with CUDA on the GPU.

The output are plots of computation time as a function of matrix size parameter D.
"""

n_samples = 30
batch_size = 512

def time_mm(D, device):
    with torch.no_grad():
        A = torch.randn(batch_size, D, device=device)
        H = build_H(D, device=device)

        # One run so we initialize things, get stable results
        ret = (H @ A.T).T

        total_time = 0
        for _ in range(n_samples):
            A = torch.randn(batch_size, D, device=device)
            t0 = time.time_ns()
            ret = (H @ A.T).T
            t1 = time.time_ns()
            dt = t1 - t0
            total_time += dt
        return total_time


def time_fwht(D, device, fwht_fun):
    with torch.no_grad():
        # One run so we initialize things, get stable results
        A = torch.randn(batch_size, D, device=device)
        torch.cuda.synchronize()
        ret = fwht_fun(A)

        t0 = time.time_ns()
        for _ in range(n_samples):
            ret = fwht_fun(A)
        t1 = time.time_ns()
        return t1 - t0


torch.manual_seed(0)
log2D_list = list(range(6, 14))

matmul_cpu_times = [time_mm(2 ** p, torch.device('cpu')) for p in log2D_list]
matmul_gpu_times = [time_mm(2 ** p, torch.device('cuda')) for p in log2D_list]
print(matmul_cpu_times)
print(matmul_gpu_times)

fwht_cpu_times = [time_fwht(2 ** p, torch.device('cpu'), cpp_fwht.FWHTFunction.apply) for p in log2D_list]
fwht_gpu_times = [time_fwht(2 ** p, torch.device('cuda'), cuda_fwht.FWHTFunction.apply) for p in log2D_list]
print(fwht_cpu_times)
print(fwht_gpu_times)

plt.figure()
plt.plot(log2D_list, matmul_cpu_times, label='matmul (CPU)')
plt.plot(log2D_list, fwht_cpu_times, label='FWHT (CPU)')
plt.plot(log2D_list, matmul_gpu_times, label='matmul (GPU)')
plt.plot(log2D_list, fwht_gpu_times, label='FWHT (GPU)')
plt.title('Regular-scale computation time comparison')
plt.legend()
plt.show()

plt.figure()
plt.plot(log2D_list, np.log1p(matmul_cpu_times), label='matmul (CPU)')
plt.plot(log2D_list, np.log1p(fwht_cpu_times), label='FWHT (CPU)')
plt.plot(log2D_list, np.log1p(matmul_gpu_times), label='matmul (GPU)')
plt.plot(log2D_list, np.log1p(fwht_gpu_times), label='FWHT (GPU)')
plt.yscale('log')
plt.title('Log-scale computation time comparison')
plt.legend()
plt.show()
