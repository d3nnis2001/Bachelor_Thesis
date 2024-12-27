from model.utils import preprocess, to_tensors, list_files
import pandas as pd
import numpy as np
from scipy.io import loadmat
import os 
from sklearn.utils import shuffle

class NearlabDatasetLoader:
    """
    NearlabDatasetLoader class to load Nearlab dataset.

    Args:
        train_paths (list): List of file paths for training data.
        test_paths (list): List of file paths for testing data.

    """
    def __init__(self, train_paths, test_paths):
        self.train_paths = train_paths
        self.test_paths = test_paths

    def load_data(self):
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []

        for train_file in self.train_paths:
            data = pd.read_csv(train_file, header=None, skiprows=[0])
            # Input Values
            X = data.iloc[:, :5120].values
            # Class
            y = data.iloc[:, 5120].values
            y = y - 1
            X = preprocess(X)
            X_train_list.append(X)
            y_train_list.append(y)
        
        for test_file in self.test_paths:
            data = pd.read_csv(test_file, header=None, skiprows=[0])
            X = data.iloc[:, :5120].values
            y = data.iloc[:, 5120].values
            y = y - 1
            X = preprocess(X)
            X_test_list.append(X)
            y_test_list.append(y)
        
        # Combine the data
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        # Convert into pytorch tensors for the model
        X_train, y_train = to_tensors(X_train, y_train)
        X_test, y_test = to_tensors(X_test, y_test)
        return X_train, y_train, X_test, y_test
    

class NinaproDatasetLoader:
    """
    NinaproDatasetLoader class to load Ninapro dataset.

    Args:
        train_paths (list): List of file paths for training data.
        test_paths (list): List of file paths for testing data.
    """
    def __init__(self, train_paths, test_paths):
        """
        Initializes the Ninapro dataset loader with training and testing file paths.

        Args:
            train_paths (list): List of file paths for training data.
            test_paths (list): List of file paths for testing data.
        """
        self.train_paths = train_paths
        self.test_paths = test_paths

    def load_data(self):
        """
        Loads and processes Ninapro data from the provided file paths.

        Returns:
            tuple: Processed training and testing data as PyTorch tensors.
        """
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []

        for train_file in self.train_paths:
            data = self._load_ninapro_file(train_file)
            X, y = self._process_data(data)
            X_train_list.append(X)
            y_train_list.append(y)

        for test_file in self.test_paths:
            data = self._load_ninapro_file(test_file)
            X, y = self._process_data(data)
            X_test_list.append(X)
            y_test_list.append(y)

        # Combine the data
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        # Convert into PyTorch tensors
        X_train, y_train = self._to_tensors(X_train, y_train)
        X_test, y_test = self._to_tensors(X_test, y_test)
        return X_train, y_train, X_test, y_test

    def _load_ninapro_file(self, file_path):
        """
        Loads a Ninapro data file.

        Args:
            file_path (str): Path to the Ninapro data file.

        Returns:
            pd.DataFrame: Loaded data as a Pandas DataFrame.
        """
        data = pd.read_csv(file_path, header=None)
        return data

    def _process_data(self, data):
        """
        Processes raw Ninapro data.

        Args:
            data (pd.DataFrame): Raw data from a Ninapro file.

        Returns:
            tuple: Processed feature matrix and labels.
        """
        # Extract input values (features) and class labels
        X = data.iloc[:, :-1].values  # All columns except the last
        y = data.iloc[:, -1].values  # The last column contains labels

        # Normalize labels (optional, subtract 1 to make labels zero-indexed)
        y = y - 1

        # Preprocess input features
        X = self._preprocess(X)
        return X, y

    def _preprocess(self, X):
        """
        Applies preprocessing to the input features.

        Args:
            X (np.ndarray): Raw input features.

        Returns:
            np.ndarray: Preprocessed features.
        """
        # Example: Normalize EMG data to range [0, 1]
        X = (X - np.min(X, axis=1, keepdims=True)) / (np.ptp(X, axis=1, keepdims=True) + 1e-8)
        return X

    def _to_tensors(self, X, y):
        """
        Converts data arrays to PyTorch tensors.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.

        Returns:
            tuple: Feature and label tensors.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return X_tensor, y_tensor