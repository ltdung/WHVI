from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import torch

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cpp', 'cuda'])
parser.add_argument('-b', '--batch-size', type=int, default=16)
parser.add_argument('-D', type=int, default=32)
parser.add_argument('-r', '--runs', type=int, default=100)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='us')
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-d', '--double', action='store_true')
options = parser.parse_args()

if options.example == 'py':
    from python.fwht import FWHT
elif options.example == 'cpp':
    from cpp.fwht import FWHT
else:
    from cuda.fwht import FWHT
    options.cuda = True

device = torch.device("cuda") if options.cuda else torch.device("cpu")
dtype = torch.float64 if options.double else torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': True}
A = torch.randn(options.batch_size, options.D, options.D, **kwargs)
fwht_module = FWHT().to(device, dtype)

# Force CUDA initialization
ret = fwht_module(A)

min_time = math.inf
total_time = 0

for _ in range(options.runs):
    fwht_module.zero_grad()

    start = time.time()
    ret = fwht_module(A)
    elapsed = time.time() - start
    min_time = min(min_time, elapsed)
    total_time += elapsed

scale = TIME_SCALES[options.scale]
min_time *= scale
forward_average = total_time / options.runs * scale

print(f'FWHT: {min_time:.3f}/{forward_average:.3f} {options.scale}')
