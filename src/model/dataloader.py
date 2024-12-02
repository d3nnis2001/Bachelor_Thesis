from model.utils import preprocess, to_tensors, list_files
import pandas as pd
import numpy as np
from scipy.io import loadmat

class NearlabDatasetLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def load_data(self):
        file_paths = list_files(self.folder_path, fileformat="csv")
        print("HELLOO")
        X_train_list, y_train_list = [], []
        
        # Load the training data
        for train_file in self.file_paths[:2]:
            data = pd.read_csv(self.folder_path + train_file, header=None, skiprows=[0])
            # Input Values
            X = data.iloc[:, :5120].values
            # Class
            y = data.iloc[:, 5120].values
            y = y - 1
            X = preprocess(X)
            X_train_list.append(X)
            y_train_list.append(y)
        
        # Combine the trainingsdata
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        
        # Load the test data
        data = pd.read_csv(self.folder_path + self.file_paths[2], header=None, skiprows=[0])
        X_test = data.iloc[:, :5120].values # Values
        y_test = data.iloc[:, 5120].values  # Class
        y_test = y_test - 1
        X_test = preprocess(X_test)
        
        # Convert into pytorch tensors for the model
        X_train, y_train = to_tensors(X_train, y_train)
        X_test, y_test = to_tensors(X_test, y_test)
        return X_train, y_train, X_test, y_test
    

class NinaproDatasetLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def load_data(self):
        mat_files = list_files(self.folder_path, fileformat="mat")
        X_list, y_list = [], []

        for filename in mat_files:
            mat_data = loadmat(filename)
            
            # Ensure required keys exist
            if 'emg' in mat_data and 'restimulus' in mat_data:
                X = mat_data['emg'] 
                y = mat_data['restimulus'].flatten()

                # Align dimensions of X and y
                min_len = min(len(X), len(y))
                X = X[:min_len]
                y = y[:min_len]
                
                # Remove datapoints where no movement is being performed
                valid_idx = y > 0
                X = X[valid_idx]
                y = y[valid_idx]
                
                X_list.append(X)
                y_list.append(y)
            else:
                print(f"Missing keys in file: {filename}")

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        return to_tensors(X, y)        