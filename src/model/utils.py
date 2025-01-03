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