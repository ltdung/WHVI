import numpy as np
import torch

from src.evaluation import evaluate_bayesian_regression_dnn

np.random.seed(0)
torch.manual_seed(0)

data = np.loadtxt('../datasets/yacht_hydrodynamics.data').astype(np.float32)
X = data[:, :-1]
y = data[:, -1].reshape(-1, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print(f'Using torch device: {device}')
error_mean, error_sd, mnll_mean, mnll_sd = evaluate_bayesian_regression_dnn(
    X, y, device, checkpoint_dir='../checkpoints/yacht'
)
