import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pathlib

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.layers import WHVILinear
from src.networks import WHVIRegression


def make_optimizer(net, gamma=0.0005, p=0.3, lambda0=0.001):
    """
    Configure the Adam optimizer as used in the experiments.

    :param net: target model.
    :param gamma: decay parameter.
    :param p: decay parameter.
    :param lambda0: learning rate.
    :return tuple: optimizer and scheduler object object.
    """
    optimizer = optim.Adam(net.parameters(), lr=lambda0)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda t: lambda0 * (
            (1 + gamma * t) ** (-p)))  # Set the learning rate decay
    return optimizer, scheduler


def evaluate_bayesian_regression_dnn(X: np.ndarray, y: np.ndarray, device, checkpoint_dir):
    """
    Compute test error (MSE) and test MNLL on the dataset (X, y).
    This is related to section 3.2 in the paper and section D.1 in the supplement.

     Details:
      * the number of neurons in the network's layers is (n_in, 128, 128, n_out = 1);
      * the network uses ReLU activations on all but the last layer;
      * data is randomly divided into 90%/10% train/test splits 8 times, we report the sample mean and sample standard
        deviation;
      * input features are standardized (in this function, not beforehand);
      * targets are unchanged;
      * we use the Adam optimizer with exponential learning rate decay (lambda0 = 0.001, p = 0.3, gamma = 0.0005);
      * training is performed for 500 steps with fixed noise variance (i.e. the variance of the gaussian likelihood in
        ELBO) and 50000 steps with optimized noise variance;
      * batch size is fixed to 64;
      * expected log likelihood (for ELBO) is estimated with 1 Monte Carlo sample during training and 64 Monte Carlo
        samples during testing.

    :param X: matrix of inputs.
    :param y: vector of targets.
    :param device: torch device.
    :param checkpoint_dir: directory where models will be saved each 5000 epochs of the 50000 epoch optimization.
    :return Tuple[float]: sample mean and sample standard deviation of the test error (MSE) and test MNLL (WHVI) across
                          8 random splits of data.
    """
    assert len(y.shape) == 2 and y.shape[1] == 1
    assert len(X.shape) == 2
    assert len(X) == len(y)

    test_errors = []
    test_mnlls = []

    # Standardize columns of X
    X = StandardScaler().fit_transform(X)

    for index in range(8):
        print(f'Iteration {index + 1}/8')

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1)

        # Create the train and test datasets
        train_dataset = TensorDataset(torch.tensor(X_train, device=device), torch.tensor(y_train, device=device))
        train_loader = DataLoader(train_dataset, batch_size=64)
        X_test = torch.tensor(X_test, device=device)
        y_test = torch.tensor(y_test, device=device)

        # Set up the model
        model = WHVIRegression([
            WHVILinear(X_test.size()[1], 128, lambda_=15.0),
            nn.ReLU(),
            WHVILinear(128, 128, lambda_=15.0),
            nn.ReLU(),
            WHVILinear(128, 1)
        ], eval_samples=64)
        model = model.to(device=device)

        # Set up the optimizer and learning rate scheduler
        optimizer, scheduler = make_optimizer(model)

        # Train the model
        iteration_dir = pathlib.Path(checkpoint_dir) / f'iter-{index}'
        iteration_dir.mkdir(exist_ok=True, parents=True)
        model.train_model(train_loader, optimizer, scheduler, epochs1=500, epochs2=50000, pbar_update_period=1,
                          checkpoint_dir=iteration_dir)

        # Evaluate the model on test data and store the result in lists
        error, mnll = model.eval_model(X_test, y_test)
        print(f"MSE: {error}, MNLL: {mnll}")
        test_errors.append(error)
        test_mnlls.append(mnll)

    # Report the sample mean and sample standard deviation of errors
    error_mean = np.mean(test_errors)
    error_sd = np.std(test_errors)
    mnll_mean = np.mean(test_mnlls)
    mnll_sd = np.std(test_mnlls)
    return error_mean, error_sd, mnll_mean, mnll_sd
