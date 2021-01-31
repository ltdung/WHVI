from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.autograd import gradcheck

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cpp', 'cuda'])
parser.add_argument('-b', '--batch-size', type=int, default=3)
parser.add_argument('-D', type=int, default=32)
parser.add_argument('-c', '--cuda', action='store_true')
options = parser.parse_args()

if options.example == 'py':
    from python.fwht import FWHTFunction
elif options.example == 'cpp':
    from cpp.fwht import FWHTFunction
else:
    from cuda.fwht import FWHTFunction

    options.cuda = True

device = torch.device("cuda") if options.cuda else torch.device("cpu")

kwargs = {'dtype': torch.float64,
          'device': device,
          'requires_grad': True}

A = torch.randn(options.batch_size, options.D, options.D, **kwargs)
variables = [A]

if gradcheck(FWHTFunction.apply, variables):
    print('Ok')
