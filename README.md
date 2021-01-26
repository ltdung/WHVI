# WHVI: Walsh-Hadamard Variational Inference for Bayesian Deep Learning

Reproduction of the paper [Walsh-Hadamard Variational Inference for Bayesian Deep Learning](https://proceedings.neurips.cc//paper/2020/hash/6df182582740607da754e4515b70e32d-Abstract.html).
This code is a work-in-progress and is unfinished as of the latest update (2021/01/20).

The repository is organized as follows:
* `src` contains the source code for the core WHVI functionality;
* `test` contains unit tests for manually implemented functions;
* `benchmarks` contains speed tests for manually implemented functions;
* `experiments` contains reproductions of the experiments from the original paper. 

## Example
The snippet below illustrates the use of a feed-forward regression network that uses WHVI layers.
It also is possible to seamlessly add standard modules like `nn.Linear` layers.

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
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda t: (1 + 0.0005 * t) ** (-0.3))

# Set up the data
train_loader = DataLoader(...)  # Create your train dataset loader here
model.train_model(train_loader, optimizer, scheduler)
x_test = torch.Tensor(...)  # Test data
y_test = torch.Tensor(...)  # Test targets

# Evaluate the model on test data
error, mnll = model.eval_model(x_test, y_test)
```

See the [Toy example](./experiments/Toy%20example.ipynb) notebook for additional information.

## Setup instructions
Type the following instructions into an appropriate terminal:
```
conda env create -f environment.yml  # Configure the main environment
cd src/fwht/cpp && python setup.py install || cd ../../..  # Compile C++ FWHT functions
```
The main dependency is PyTorch.
We use Sklearn and Numpy for to evaluate models on standard datasets.
We show progress bars during model training using tqdm.

## References
* S. Rossi, S. Marmin, and M. Filippone (2019). [Walsh-Hadamard Variational Inference for Bayesian Deep Learning.](https://arxiv.org/abs/1905.11248) *arXiv preprint arXiv:1905.11248*  
* S. Rossi, S. Marmin, and M. Filippone (2019). \[Direct link\] [Supplementary material for "Walsh-Hadamard Variational Inference for Bayesian Deep Learning"](https://www.eurecom.fr/fr/publication/6398/download/data-publi-6398.pdf)
* [HazyResearch/structured-nets repository](https://github.com/HazyResearch/structured-nets/tree/master/pytorch) (source code for the Fast Walsh-Hadamard transform CUDA kernel)