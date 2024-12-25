from model.utils import preprocess, to_tensors, list_files
import pandas as pd
import numpy as np
from scipy.io import loadmat
import os 
from sklearn.utils import shuffle

class NearlabDatasetLoader:
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
    def __init__(self, train_paths, test_paths):
        self.train_paths = train_paths
        self.test_paths = test_paths
        self.window_length = 400
        self.window_increment = 40

    def get_windows(self, data):
        """Create windows from the continuous data"""
        windows = []
        for i in range(0, len(data) - self.window_length + 1, self.window_increment):
            windows.append(data[i:i + self.window_length])
        return np.array(windows)

    def load_data(self):
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []
        
        # Load training files
        for train_file in self.train_paths:
            data = pd.read_csv(train_file)
            # Assume last column is label
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            
            # Create windows and reshape
            X = self.get_windows(X)
            X = X.transpose(0, 2, 1)  # to (n_samples, n_channels, n_timepoints)
            
            # Extend y to match window length
            y = np.repeat(y[self.window_length-1:], 
                         len(range(0, len(data) - self.window_length + 1, self.window_increment)))
            
            X_train_list.append(X)
            y_train_list.append(y)
        
        # Load test files
        for test_file in self.test_paths:
            data = pd.read_csv(test_file)
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            
            X = self.get_windows(X)
            X = X.transpose(0, 2, 1)
            
            y = np.repeat(y[self.window_length-1:], 
                         len(range(0, len(data) - self.window_length + 1, self.window_increment)))
            
            X_test_list.append(X)
            y_test_list.append(y)
        
        # Combine the data
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        # Shuffle the data
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)
        
        # Convert to PyTorch tensors
        X_train, y_train = to_tensors(X_train, y_train)
        X_test, y_test = to_tensors(X_test, y_test)
        
        return X_train, y_train, X_test, y_test

    def set_window_params(self, window_length=400, window_increment=40):
        """Allow customization of window parameters"""
        self.window_length = window_length
        self.window_increment = window_increment