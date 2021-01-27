# WHVI: Walsh-Hadamard Variational Inference for Bayesian Deep Learning

Reproduction of the paper [Walsh-Hadamard Variational Inference for Bayesian Deep Learning](https://proceedings.neurips.cc//paper/2020/hash/6df182582740607da754e4515b70e32d-Abstract.html).
This code is a work-in-progress and is unfinished as of the latest update (2021/01/20).

The repository is organized as follows:
* `src` contains the source code for the core WHVI functionality;
* `test` contains unit tests for manually implemented functions;
* `benchmarks` contains speed tests for manually implemented functions;
* `experiments` contains reproductions of the experiments from the original paper;
* `report` contains a reproducibility report;

## Example
The snippet below illustrates the use of a feed-forward regression network that uses WHVI layers.
It also is possible to seamlessly add standard modules like `nn.Linear` layers.
See the [Toy example](./experiments/Toy%20example.ipynb) notebook for additional information.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from networks import WHVIRegression
from layers import WHVILinear

# Seed for reproducibility
torch.manual_seed(0)

# Use GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up the data
x = torch.randn(200, 3, device=device)
y = torch.reshape(x[:, 0] + x[:, 1] ** 2 - 0.3 * x[:, 2] ** 3, (-1, 1))
perm = torch.randperm(200)
x_train, x_test = x[perm[:150]], x[perm[150:]]
y_train, y_test = y[perm[:150]], y[perm[150:]]
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64)

# Create the model and optimization objects
model = WHVIRegression([
    WHVILinear(3, 16, lambda_=2.0),
    torch.nn.ReLU(),
    WHVILinear(16, 1)
])
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: (1 + 0.0005 * t) ** (-0.3))
model.train_model(train_loader, optimizer, scheduler, epochs1=500, epochs2=1500)

# Evaluate the model on test data
error, mnll = model.eval_model(x_test, y_test)
```

## Setup instructions
Type the following commands into a terminal:
```
conda env create -f environment.yml                         # Create the conda environment
cd src/fwht/cpp && python setup.py install || cd ../../..   # Compile C++ FWHT functions
cd src/fwht/cuda && python setup.py install || cd ../../..  # Compile CUDA kernel for FWHT
```
The main dependency is PyTorch.
We use Sklearn and Numpy for to evaluate models on standard datasets.
We show progress bars during model training using tqdm.

## References
* S. Rossi, S. Marmin, and M. Filippone (2019). [Walsh-Hadamard Variational Inference for Bayesian Deep Learning.](https://arxiv.org/abs/1905.11248) *arXiv preprint arXiv:1905.11248*  
* S. Rossi, S. Marmin, and M. Filippone (2019). \[Direct link\] [Supplementary material for "Walsh-Hadamard Variational Inference for Bayesian Deep Learning"](https://www.eurecom.fr/fr/publication/6398/download/data-publi-6398.pdf)
* [HazyResearch/structured-nets repository](https://github.com/HazyResearch/structured-nets/tree/master/pytorch) (source code for the Fast Walsh-Hadamard transform CUDA kernel)