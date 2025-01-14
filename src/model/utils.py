import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler

def preprocess(X):
    """
    Preprocesses the input data.

    Parameters:
    -----------
    X : numpy.ndarray
        Input data
    """
    # Reshape X to (n_samples, n_channels, n_timepoints) for more information look into next illustration
    X = X.reshape(-1, 10, 512)
    # Standardization (Z-Score Normalization in this case)
    standard = StandardScaler()
    # Scales each channel independently from each other
    X = np.array([standard.fit_transform(channel.T).T for channel in X])
    return X

def to_tensors(X, y):
    """
    Converts numpy arrays to PyTorch tensors.

    Parameters:
    -----------
    X : numpy.ndarray
        Input data
    y : numpy.ndarray
        Target labels
    """
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    return X_tensor, y_tensor

def list_files(directory, fileformat):
    """
    Lists all files in a directory with a specific file format.

    Parameters:
    -----------
    directory : str
        Directory to search for files.
    fileformat : str
        File format to search for.
    """
    found_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(fileformat):
                found_files.append(os.path.join(root, file).replace("\\", "/"))
    return found_files

def take_n_shots(X, y, n_shots, y_target):
    """
    Take n_shots from the target class.

    Parameters:
    -----------
    X : numpy.ndarray
        Input data
    y : numpy.ndarray
        Target labels
    n_shots : int
        Number of shots to take
    y_target : int
        Target class
    """
    mask = y == y_target
    X_target = X[mask]
    y_target = y[mask]
    # Select n_shots from the target class randomly
    indices = torch.randperm(X_target.size(0))[:n_shots]
    X_shots = X_target[indices]
    y_shots = y_target[indices]
    return X_shots, y_shots

def take_n_samples_from_every_class(X, y, n_samples):
    """
    Take n_samples from every class.

    Parameters:
    -----------
    X : numpy.ndarray
        Input data
    y : numpy.ndarray
        Target labels
    n_samples : int
        Number of samples to take from each class
    """
    X_samples = []
    y_samples = []
    # Loop through all classes
    for i in torch.unique(y):
        mask = y == i
        X_class = X[mask]
        y_class = y[mask]
        indices = torch.randperm(X_class.size(0))[:n_samples]
        X_sample = X_class[indices]
        y_sample = y_class[indices]
        X_samples.append(X_sample)
        y_samples.append(y_sample)
    return torch.cat(X_samples, dim=0), torch.cat(y_samples, dim=0)