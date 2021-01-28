import os
import numpy as np
import pandas as pd
import torch
import pathlib
import requests
import zipfile

from src.evaluation import evaluate_bayesian_regression_dnn

np.random.seed(0)
torch.manual_seed(0)

dataset_path = pathlib.Path('../datasets') / 'naval.csv'
zip_dataset_path = pathlib.Path('../datasets') / 'naval.zip'
if not dataset_path.exists():
    print('Downloading Naval')

    r = requests.get('http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip', stream=True)
    with open(str(zip_dataset_path), 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)

    zip = zipfile.ZipFile(str(zip_dataset_path))
    data = np.array([list(map(float, l.split())) for l in zip.read(r'UCI CBM Dataset/data.txt').splitlines()])
    df = pd.DataFrame(data)
    df.to_csv(str(dataset_path), index=False)
else:
    df = pd.read_csv(dataset_path)

X = df.values[:, :-2].astype(np.float32)
y = df.values[:, -2:].astype(np.float32).reshape(-1, 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print(f'Using torch device: {device}')
error_mean, error_sd, mnll_mean, mnll_sd = evaluate_bayesian_regression_dnn(
    X, y, device, checkpoint_dir='../checkpoints/kin8nm'
)
