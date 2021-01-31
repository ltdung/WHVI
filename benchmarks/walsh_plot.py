import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import build_H

import src.fwht.cpp.fwht as cpp_fwht
import src.fwht.cuda.fwht as cuda_fwht
import src.fwht.python.fwht_baseline as python_fwht

"""
Measure speed of matrix multiplication and fast Walsh-Hadamard transform (Python and C++ implementations) on the CPU.
Measure speed of matrix multiplication and fast Walsh-Hadamard transform with CUDA on the GPU.

The output are plots of computation time as a function of matrix size parameter D.
"""

torch.manual_seed(0)
n_samples = 1
n_features = 64
batch_size = 512
log2D_list = list(range(6, 14))

# CPU test
device = torch.device('cpu')
cpu_matmul_times = []
cpu_fwht_python_times = []

for p in log2D_list:
    print(p)
    D = 2 ** p
    A = torch.randn(batch_size, D, n_features, device=device)
    H = build_H(D, device)

    # Matrix multiplication
    matmul_time = 0
    for _ in range(n_samples):
        t0 = time.time()
        torch.stack([H @ A[i] for i in range(batch_size)])
        matmul_time += time.time() - t0
    matmul_time /= n_samples

    python_time = 0
    for _ in range(n_samples):
        t0 = time.time()
        python_fwht.FWHTFunction.apply(A)
        python_time += time.time() - t0
    python_time /= n_samples

    cpu_fwht_python_times.append(python_time)
    cpu_matmul_times.append(matmul_time)

# The python and C++ implementations have the same speed

# GPU test
device = torch.device('cuda')
gpu_matmul_times = []
gpu_fwht_cuda_times = []

# Initialize CUDA
torch.randn(2, 2, device=device) @ torch.randn(2, 2, device=device)

for p in log2D_list:
    print(p)
    D = 2 ** p
    A = torch.randn(batch_size, D, n_features, device=device)  # batch_size vectors, each of size D
    H = build_H(D, device)

    # Matrix multiplication
    matmul_time = 0
    for _ in range(n_samples):
        t0 = time.time()
        torch.stack([H @ A[i] for i in range(batch_size)])
        matmul_time += time.time() - t0
    matmul_time /= n_samples

    cuda_time = 0
    for _ in range(n_samples):
        t0 = time.time()
        cpp_fwht.FWHTFunction.apply(A)
        cuda_time += time.time() - t0
    cuda_time /= n_samples

    gpu_fwht_cuda_times.append(cuda_time)
    gpu_matmul_times.append(matmul_time)

plt.figure()
plt.plot(log2D_list, cpu_matmul_times, label='matmul (CPU)')
plt.plot(log2D_list, cpu_fwht_python_times, label='FWHT (CPU)')
plt.plot(log2D_list, gpu_matmul_times, label='matmul (GPU)')
plt.plot(log2D_list, gpu_fwht_cuda_times, label='FWHT (GPU)')
plt.title('Regular-scale computation time comparison')
plt.legend()
plt.show()

plt.figure()
plt.plot(log2D_list, np.log1p(cpu_matmul_times), label='matmul (CPU)')
plt.plot(log2D_list, np.log1p(cpu_fwht_python_times), label='FWHT (CPU)')
plt.plot(log2D_list, np.log1p(gpu_matmul_times), label='matmul (GPU)')
plt.plot(log2D_list, np.log1p(gpu_fwht_cuda_times), label='FWHT (GPU)')
plt.yscale('log')
plt.title('Log-scale computation time comparison')
plt.legend()
plt.show()
