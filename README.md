# WHVI: Walsh-Hadamard Variational Inference for Bayesian Deep Learning

Code for the paper [Walsh-Hadamard Variational Inference for Bayesian Deep Learning](https://proceedings.neurips.cc//paper/2020/hash/6df182582740607da754e4515b70e32d-Abstract.html).
This code is a work-in-progress and is unfinished as of the latest update (2021/01/19).

The repository is organized as follows:
* `src` contains the source code for WHVI layers, neural network examples which use the layers, utility functions for fast matrix operations and for the fast Walsh-Hadamard transform (FWHT);
* `test` contains unit tests for manually implemented functions;
* `speed_benchmarks` contains speed tests for manually implemented functions;
* `experiments` contains reproductions of the experiments from the original paper. 

## Example
The snippet below illustrates the use of a feed-forward regression network that uses WHVI layers.
It is possible to add standard layers like `nn.Linear` seamlessly.

```python
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from networks import WHVIRegression
from layers import WHVILinear

# Create the network and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WHVIRegression([
    WHVILinear(8, 128, device=device),
    nn.ReLU(),
    WHVILinear(128, 128, device=device),
    nn.ReLU(),
    WHVILinear(128, 1, device=device)
])
model = model.to(device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
optim.lr_scheduler.LambdaLR(optimizer, lambda t: (1 + 0.0005 * t) ** (-0.3))

# Set up the data
train_loader = DataLoader(...)  # Create your train dataset loader here
model.train_model(train_loader, optimizer)
X_test = torch.Tensor(...)  # Test data
y_test = torch.Tensor(...)  # Test targets

# Evaluate the model on test data
error, mnll = model.eval_model(X_test, y_test)
```