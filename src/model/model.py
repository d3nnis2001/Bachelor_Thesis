import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model.layer import GLVQ, GMLVQ
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import os
import numpy as np
import pandas as pd

class CNet2D(nn.Module):
    """
    Convolutional Neural Network for 2D data. The network consists of 3 convolutional blocks followed by a dense layer.
    The final layer is either a softmax layer for classification or a GLVQ/GMLVQ layer for prototype-based learning.

    Parameters:
    -----------
    version : str
        Version of the model. Can be either "Softmax", "GLVQ" or "GMLVQ".
    num_prototypes_per_class : int
        Number of prototypes per class for the prototype-based models.
    num_classes : int
        Number of classes in the dataset.
    epochs : int
        Number of epochs for training.
    optimizer_type : str
        Type of optimizer
    learning_rate : float
        Learning rate for the optimizer.
    batch_size : int
        Batch size for training.
    device : torch.device
        Device to use for training. If None, the device is automatically selected based on availability.
    
    """
    def __init__(self, version="GLVQ", num_prototypes_per_class=1, num_classes=8, epochs=5, optimizer_type="ADAM", learning_rate=0.001, batch_size=128, device=None, dataset_type="NearLab"):
        super(CNet2D, self).__init__()
        # Parameters
        self.version = version
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_classes = num_classes
        self.epochs = epochs
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Check if cuda available
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.feature_extractor = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 32, kernel_size=(3, 13), padding=(1, 6)),
            nn.BatchNorm2d(32),
            nn.RReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.3),
            
            # Conv Block 2
            nn.Conv2d(32, 48, kernel_size=(3, 9), padding=(1, 4)),
            nn.BatchNorm2d(48),
            nn.RReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.3),
            
            # Conv Block 3
            nn.Conv2d(48, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.RReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.3)
        )

        if dataset_type == "NinaPro":
            flattened_size = 49152
        elif dataset_type == "NearLab":
            flattened_size = 40960
        
        # Dense layers
        self.dense_features = nn.Sequential(
            nn.Linear(flattened_size, 300),
            nn.BatchNorm1d(300),
            nn.RReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(300, 50),
            nn.BatchNorm1d(50),
            nn.RReLU()
        )
        
        # Final classification layer
        if self.version == "Softmax":
            self.classifier = nn.Linear(50, num_classes)
            # GLVQ layer
        elif self.version == "GLVQ":
            self.classifier = GLVQ(50, self.num_prototypes_per_class, self.num_classes)
        else:
            # GMLVQ layer
            self.classifier = GMLVQ(50, self.num_prototypes_per_class, self.num_classes)

        self.to(self.device)

    def extract_features(self, x):
        """
        Extracts features from input data

        Parameters:
        -----------
        x : torch.Tensor
            Input data
        """
        x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1) 
        features = self.dense_features(x)
        return features
    
    def forward(self, x, y=None):
        """
        Forward pass through the network

        Parameters:
        -----------
        x : torch.Tensor
            Input data
        """
        # Move data to device
        x = x.to(self.device)
        if y is not None:
            y = y.to(self.device)

        # Extract features
        features = self.extract_features(x)
        # Return the output based on the model version
        if self.version == "Softmax":
            return F.log_softmax(self.classifier(features), dim=1)
        else:
            return self.classifier(features, y)
    
    def fit(self, X, y, validation_split=0.2, patience=10, min_delta=0.001):
        """
        Fits the model to the input train data X and y

        Parameters:
        -----------
        X : torch.Tensor
            Input data
        """
        # Trainingsmode
        self.train()
        X, y = X.to(self.device), y.to(self.device)
        validation_size = int(validation_split * len(X))
        training_size = len(X) - validation_size

        _, val_X = torch.split(X, [training_size, validation_size])
        _, val_y = torch.split(y, [training_size, validation_size])
        # Define validation and test loader
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(val_X, val_y)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Check which optimizer to use
        optimizer = (optim.Adam if self.optimizer_type == "ADAM" else optim.SGD)(
            self.parameters(), lr=self.learning_rate
        )
        
        # Variables for early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Keep track of the training history
        history = {
            "loss": [],
            "val_loss": [],
            "epoch": []
        }

        for epoch in range(self.epochs):
            epoch_losses = []
            # Go through the batches
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                if self.version in ["GLVQ", "GMLVQ"]:
                    loss = self(batch_X, batch_y)
                else:  # Softmax version
                    outputs = self(batch_X)
                    loss = F.nll_loss(outputs, batch_y)

                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            
            # Validation
            self.eval()
            val_losses = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    if self.version in ["GLVQ", "GMLVQ"]:
                        val_loss = self(batch_X, batch_y)
                    else:
                        outputs = self(batch_X)
                        val_loss = F.nll_loss(outputs, batch_y)
                    val_losses.append(val_loss.item())

            avg_loss = np.mean(epoch_losses)
            avg_val_loss = np.mean(val_losses)

            history["loss"].append(avg_loss)
            history["epoch"].append(epoch + 1)
            history["val_loss"].append(avg_val_loss)
            
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

            # Early stopping check
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
    
        # Load the best model
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        return history
                    
    def predict(self, X):
        """
        Predicts the class labels for the input data

        Parameters:
        -----------
        X : torch.Tensor
            Input data
        """
        self.eval()
        X = X.to(self.device)
        with torch.no_grad():
            if self.version == "Softmax":
                # Predict class labels with argmax
                outputs = self(X)
                return torch.argmax(outputs, dim=1)
            else:
                # Extract features and predict
                features = self.extract_features(X)
                return self.classifier.predict(features)
            
    def add_new_class(self, new_data, new_labels):
        """
        Adds a new class to the model
        
        Parameters:
        -----------
        new_data : torch.Tensor
            Data of new class
        new_labels : torch.Tensor
            Label of new class
        num_prototypes : int
            Number of prototypes for the class
        """
        if self.version in ["GLVQ", "GMLVQ"]:
            with torch.no_grad():
                self.num_classes += 1
                new_data = new_data.to(self.device)
                new_labels = new_labels.to(self.device)
                # Extract features
                features = self.extract_features(new_data)
                # Calc mean of training data
                class_mean = features.mean(dim=0)
                # Repeat it incase we have multiple prototypes
                prototype_features = class_mean.repeat(self.num_prototypes_per_class, 1)
                prototype_labels = new_labels.repeat(self.num_prototypes_per_class)
                # Adds the Prototype to the layer
                self.classifier.add_prototypes(prototype_features, prototype_labels)
        else:
            # Incase you try to use it with the softmax version
            raise ValueError("Prototype addition is only supported for GLVQ or GMLVQ versions.")
        
    def optimize_new_prototypes(self, new_data, new_labels, epochs=5):
        """
        Optimizes the new prototypes for few-shot learning
        
        Parameters:
        -----------
        new_data : torch.Tensor
            Data of the new class
        new_labels : torch.Tensor
            Labels of the new class
        epochs : int
            Number of optimization steps
        """
        if self.version in ["GLVQ", "GMLVQ"]:
            new_data = new_data.to(self.device)
            new_labels = new_labels.to(self.device)
            
            # Create optimizer for all prototypes
            optimizer = (optim.Adam if self.optimizer_type == "ADAM" else optim.SGD)([self.classifier.prototypes], lr=self.learning_rate)
            # Create a temporary dataset and loader for batch processing
            train_dataset = TensorDataset(new_data, new_labels)
            train_loader = DataLoader(train_dataset, 
                                    batch_size=min(32, len(new_data)), 
                                    shuffle=True)
            
            # Store original prototypes
            with torch.no_grad():
                original_prototypes = self.classifier.prototypes.clone()
                start_idx = (self.num_classes - 1) * self.num_prototypes_per_class
            
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    features = self.extract_features(batch_X)
                    loss = self.classifier(features, batch_y)
                    loss.backward()
                    
                    # Only update new prototypes, reset old ones
                    with torch.no_grad():
                        self.classifier.prototypes.data[:start_idx] = original_prototypes[:start_idx].clone()
                    
                    optimizer.step()
                    epoch_loss += loss.item()
                
                print(f"FSL Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        else:
            raise ValueError("Prototype optimization is only supported for GLVQ or GMLVQ versions.")
            
    def evaluate_model(self, X, y, conf_matrix=True, sub_acc=True):
        """
        Evaluates the model with multiple metrics.
        
        Parameters:
        -----------
        X : torch.Tensor
            Test data
        y : torch.Tensor
            Test labels
        conf_matrix : bool
            Whether to plot confusion matrix
        sub_acc : bool
            Whether to print detailed accuracy metrics
                
        """
        # Set eval mode
        self.eval()
        X = X.to(self.device)
        y = y.to(self.device)
        
        with torch.no_grad():
            predicted = self.predict(X)
        
        # Convert to numpy for sklearn metrics
        y_true = y.cpu().tolist()
        y_pred = predicted.cpu().tolist()
        
        # Calculate accuracy
        accuracy = (predicted == y).float().mean()
        print(f"Test Accuracy: {accuracy.item():.4f}")
        
        # Define class names
        class_names = ["Flexion", "Extension", "Supination", "Pronation", 
                    "Open", "Pinch", "Lateral pinch", "Fist"]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if conf_matrix:
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names,
                        yticklabels=class_names)
            plt.title(f"Confusion Matrix - {self.version}")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"confusion_matrix_{self.version}_{timestamp}.png")
            plt.show()
        
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        if sub_acc:
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=class_names))
            
            print("\nPer-class Accuracy:")
            for name, acc in zip(class_names, per_class_acc):
                print(f"{name}: {acc:.4f}")
        
        return {
            "accuracy": accuracy.item(),
            "confusion_matrix": cm,
            "per_class_accuracy": per_class_acc,
            "predictions": y_pred,
            "true_labels": y_true
        }
    
    def save_model_state(self, save_path):
        """
        Saves the model state to the specified directory.

        Parameters:
        -----------
        save_path : str
            Path to the directory where the model state will be saved.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_file = os.path.join(save_path, "model_state.pth")
        torch.save(self.state_dict(), save_file)
        print(f"Model state saved to {save_file}")
    
    def save_history_csv(self, history, save_path):
        """
        Saves the training history to a CSV file.

        Parameters:
        -----------
        history : dict
            Training history
        save_path : str
            Path to the directory where the CSV file will be saved.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df = pd.DataFrame(history)
        save_file = os.path.join(save_path, "training_history.csv")
        df.to_csv(save_file, index=False)
        print(f"Training history saved to {save_file}")


    def load_model_state(self, path):
        """
        Loads in a state of the model.

        Parameters:
        -----------
        path : str
            Path to th directory where the pth file is.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        self.load_state_dict(torch.load(path))
        self.eval()