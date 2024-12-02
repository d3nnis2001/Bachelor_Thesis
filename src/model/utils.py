import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler

def preprocess(X):
    # Reshape X to (n_samples, n_channels, n_timepoints) for more information look into next illustration
    X = X.reshape(-1, 10, 512)
    # Standardization (Z-Score Normalization in this case)
    standard = StandardScaler()
    # Scales each channel independently from each other
    X = np.array([standard.fit_transform(channel.T).T for channel in X])
    return X

def to_tensors(X, y):
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    return X_tensor, y_tensor

def list_files(directory, fileformat):
    files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(fileformat):
                files.append(os.path.join(root, file).replace("\\", "/"))
    return files