from model.utils import preprocess, to_tensors, list_files
import pandas as pd
import numpy as np
import torch
from scipy.io import loadmat
import os 
from sklearn.utils import shuffle

class NearlabDatasetLoader:
    """
    NearlabDatasetLoader class to load Nearlab dataset.

    Parameters:
    ----------
    train_paths : list
        List of file paths for training data.
    test_paths : list
        List of file paths for testing data.

    """
    def __init__(self, train_paths, test_paths):
        self.train_paths = train_paths
        self.test_paths = test_paths

    def load_data(self, split_method="file_split"):
        if split_method == "file_split":
            return self.split_by_file()
        elif split_method == "repetition_wise":
            return self.split_data_by_repetitions()

    def _read_in_file(self, filepath):
        data = pd.read_csv(filepath, header=None, skiprows=[0])
        X = data.iloc[:, :5120].values
        y = data.iloc[:, 5120].values
        repetitions = data.iloc[:, 5121].values
        y = y - 1
        X = preprocess(X)
        return X, y, repetitions
    
    def split_by_file(self):
        """
        Splits by rotation. We use 2 rotations for training and one for testing. Ensures generalization across all rotations
        """
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []

        for train_file in self.train_paths:
            X_train, y_train, _ = self._read_in_file(train_file)
            X_train_list.append(X_train)
            y_train_list.append(y_train)
        
        for test_file in self.test_paths:
            X_test, y_test, _ = self._read_in_file(test_file)
            X_test_list.append(X_test)
            y_test_list.append(y_test)
        
        # Combine the data
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        # Convert into pytorch tensors for the model
        X_train, y_train = to_tensors(X_train, y_train)
        X_test, y_test = to_tensors(X_test, y_test)
        return X_train, y_train, X_test, y_test

    def split_data_by_repetitions(self):
        """
        Splits the data by taking 3/5 of repetitions for training and 2/5 for testing
        for each movement in each hand orientation
        """
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []
        
        # Process all files
        for filepath in self.train_paths + self.test_paths:
            X, y, repetitions = self._read_in_file(filepath)
            
            # For each movement class
            for movement in np.unique(y):
                # Get all indices for the current movement
                movement_indices = np.where(y == movement)[0]
                # Get all repetitions for the current movement
                movement_repetitions = repetitions[movement_indices]
                
                # Get all unique repetitions
                unique_reps = np.unique(movement_repetitions)
                # Number of repetitions
                num_reps = len(unique_reps)
                # Take 3/5 of repetitions for training
                num_train_reps = int(np.ceil(3 * num_reps / 5))
                
                # Randomly select repetition numbers for training
                train_rep_nums = np.random.choice(unique_reps, size=num_train_reps, replace=False)
                
                # Split data based on the selected repetition numbers
                train_indices = movement_indices[np.isin(movement_repetitions, train_rep_nums)]
                test_indices = movement_indices[~np.isin(movement_repetitions, train_rep_nums)]
                
                X_train_list.append(X[train_indices])
                y_train_list.append(y[train_indices])
                X_test_list.append(X[test_indices])
                y_test_list.append(y[test_indices])
        
        # Combine all data
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        # Shuffle the data
        X_train, y_train = shuffle(X_train, y_train, random_state=39)
        X_test, y_test = shuffle(X_test, y_test, random_state=39)
        
        X_train, y_train = to_tensors(X_train, y_train)
        X_test, y_test = to_tensors(X_test, y_test)
        
        return X_train, y_train, X_test, y_test

    

class NinaproDatasetLoader:
    """
    NinaproDatasetLoader class to load Ninapro dataset.

    Parameters:
    ----------
    train_paths : list
        List of file paths for training data.
    test_paths : list
        List of file paths for testing data.

    """
    def __init__(self, train_paths, test_paths):

        self.train_paths = train_paths
        self.test_paths = test_paths

    def load_data(self):
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
        data = pd.read_csv(file_path, header=None)
        return data

    def _process_data(self, data):
        # Extract input values (features) and class labels
        X = data.iloc[:, :-1].values  # All columns except the last
        y = data.iloc[:, -1].values  # The last column contains labels

        # Normalize labels (optional, subtract 1 to make labels zero-indexed)
        y = y - 1

        # Preprocess input features
        X = self._preprocess(X)
        return X, y

    def _preprocess(self, X):
        # Example: Normalize EMG data to range [0, 1]
        X = (X - np.min(X, axis=1, keepdims=True)) / (np.ptp(X, axis=1, keepdims=True) + 1e-8)
        return X

    def _to_tensors(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return X_tensor, y_tensor