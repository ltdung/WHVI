import numpy as np
import pandas as pd
import torch
import pathlib

from src.evaluation import evaluate_bayesian_regression_dnn

np.random.seed(0)
torch.manual_seed(0)

dataset_path = pathlib.Path('../datasets') / 'kin8nm.csv'
if not dataset_path.exists():
    print('Downloading KIN8NM')
    df = pd.read_csv('https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff')
    df.to_csv(str(dataset_path), index=False)
else:
    df = pd.read_csv(dataset_path)

X = df.values[:, :-1].astype(np.float32)
y = df.values[:, -1].astype(np.float32).reshape(-1, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print(f'Using torch device: {device}')
error_mean, error_sd, mnll_mean, mnll_sd = evaluate_bayesian_regression_dnn(
    X, y, device, checkpoint_dir='../checkpoints/kin8nm'
)
