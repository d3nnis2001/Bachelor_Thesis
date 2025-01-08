import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from model.layer import GLVQ, GMLVQ
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import os
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
        
        if dataset_type == "NinaPro":
            self.pool = nn.MaxPool2d((1, 3))
        else:
            self.pool = nn.MaxPool2d((1, 4))
    
        self.feature_extractor = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 32, kernel_size=(3, 13), padding=(1, 6)),
            nn.BatchNorm2d(32),
            nn.RReLU(),
            self.pool,
            nn.Dropout(0.3),
            
            # Conv Block 2
            nn.Conv2d(32, 48, kernel_size=(3, 9), padding=(1, 4)),
            nn.BatchNorm2d(48),
            nn.RReLU(),
            self.pool,
            nn.Dropout(0.3),
            
            # Conv Block 3
            nn.Conv2d(48, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.RReLU(),
            self.pool,
            nn.Dropout(0.3)
        )
        # TODO: Make this dynamic instead of hardcoded
        if dataset_type == "NinaPro":
            flattened_size = 5120
        elif dataset_type == "NearLab":
            flattened_size = 5120
        
        # Dense layers
        self.dense_features = nn.Sequential(
            nn.Linear(flattened_size, 300),
            nn.BatchNorm1d(300),
            nn.RReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(300, 50),
            nn.BatchNorm1d(50),
            nn.RReLU(),
        )
        # Final classification layer
        if self.version == "Softmax":
            nn.Linear(50, num_classes)
            self.classifier = nn.Softmax(dim=1)

            
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
    
    def forward(self, x, y=None, t_value=None):
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
            return self.classifier(features)
        else:
            return self.classifier(features, y, t_value)
    
    def fit(self, X, y, patience=10, X_val = None, y_val = None):
        # TODO: Implement Learning rate scheduler maybe? With decreasing learning rate from 0.0002 to 0.00005
        """
        Fits the model to the input train data X and y

        Parameters:
        -----------
        X : torch.Tensor
            Input data
        y : torch.Tensor
            Target labels
        patience : int
            Number of epochs to wait before early stopping
        X_val : torch.Tensor
            Validation data
        y_val : torch.Tensor
            Validation labels
        """
        X, y = X.to(self.device), y.to(self.device)

        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
        # Define validation and test loader
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Check which optimizer to use
        optimizer = (optim.Adam if self.optimizer_type == "ADAM" else optim.SGD)(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Variables for early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        
        # Keep track of the training history
        history = { "loss": [], "val_loss": [], "epoch": [] }

        for epoch in range(self.epochs):
            self.train()
            running_loss = 0
            epoch_losses = []
            # Go through the batches
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                if self.version in ["GLVQ", "GMLVQ"]:
                    loss = self.forward(batch_X, batch_y, t_value = epoch)
                else:  # Softmax version
                    outputs = self.forward(batch_X)
                    loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_X.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_losses.append(epoch_loss)
            
            # Validation
            self.eval()
            val_losses = []
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    if self.version in ["GLVQ", "GMLVQ"]:
                        val_loss = self.forward(batch_X, batch_y, t_value = epoch)
                    else:
                        outputs = self.forward(batch_X)
                        val_loss = criterion(outputs, batch_y)
                    val_losses.append(val_loss.item())
                    val_loss += loss.item() * batch_X.size(0)
                val_loss = val_loss / len(val_loader)
                val_losses.append(val_loss)
            # Used torch to mean the loss since I had problems with numpy
            te = torch.tensor(epoch_losses, device=self.device)
            tv = torch.tensor(val_losses, device=self.device)

            avg_loss = torch.mean(te)
            avg_val_loss = torch.mean(tv)

            history["loss"].append(avg_loss)
            history["epoch"].append(epoch + 1)
            history["val_loss"].append(avg_val_loss)
            
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
            print(f"Validation loss: {avg_val_loss}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.state_dict(), "best_model.pth")
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
            
    def add_new_class(self, new_data):
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
                new_data = new_data.to(self.device)
                # Extract features
                features = self.extract_features(new_data)
                # Calc mean of training data
                class_mean = features.mean(dim=0)
                # Repeat it incase we have multiple prototypes
                prototype_features = class_mean.repeat(self.num_prototypes_per_class, 1)

                new_class_index = self.classifier.num_classes

                prototype_labels = torch.full(self.num_prototypes_per_class, new_class_index, dtype=torch.long, device=self.device)
                # Adds the Prototype to the layer
                self.classifier.add_prototypes(prototype_features, prototype_labels)
        else:
            with torch.no_grad():
                # Get the new class index
                new_class_index = self.classifier.num_classes
                # Get the number of features
                num_features = self.classifier.weight.size(1)
                # Initialize new weights and bias
                new_weights = torch.randn(1, num_features, device=self.device) * 0.01
                new_bias = torch.zeros(1, device=self.device)
                # Concatenate the new weights and bias with the existing ones
                self.classifier.weight = torch.nn.Parameter(torch.cat([self.classifier.weight, new_weights], dim=0))
                self.classifier.bias = torch.nn.Parameter(torch.cat([self.classifier.bias, new_bias], dim=0))
                # Update the number of classes
                self.classifier.num_classes += 1
        
    def optimize_new_prototypes(self, new_data, epochs=5):
        """
        Optimizes the new prototypes for few-shot learning.
        
        Parameters:
        -----------
        new_data : torch.Tensor
            Data of the new class.
        epochs : int
            Number of optimization steps.
        """
        if self.version in ["GLVQ", "GMLVQ"]:
            new_data = new_data.to(self.device)
            
            # Get the new class index
            new_class_index = self.classifier.num_classes 
            
            # Select the new prototypes
            new_prototypes_start = (new_class_index - 1) * self.num_prototypes_per_class
            new_prototypes_end = new_class_index * self.num_prototypes_per_class
            new_prototypes = self.classifier.prototypes[new_prototypes_start:new_prototypes_end]
            
            # Create optimizer for the new prototypes only
            optimizer = (optim.Adam if self.optimizer_type == "ADAM" else optim.SGD)([new_prototypes], lr=self.learning_rate)
            
            train_labels = torch.full(
                (new_data.size(0),), 
                new_class_index, 
                dtype=torch.long, 
                device=self.device
            )
            train_dataset = TensorDataset(new_data, train_labels)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=min(32, len(new_data)), 
                shuffle=True
            )
            # Train the new prototypes
            for epoch in range(epochs):
                epoch_loss = 0
                self.classifier.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    features = self.extract_features(batch_X)
                    loss = self.classifier(features, batch_y, t_value=epoch)
                    loss.backward()
                    
                    optimizer.step()
                    epoch_loss += loss.item()
                
                print(f"FSL Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        else:
            # Optimizer
            optimizer = (optim.Adam if self.optimizer_type == "ADAM" else optim.SGD)(self.parameters(), lr=self.learning_rate)
            # Set training labels to the new class
            train_labels = torch.full(
                (new_data.size(0),), 
                self.classifier.num_classes - 1, 
                dtype=torch.long, 
                device=self.device
            )
            # New data and labels
            train_dataset = TensorDataset(new_data, train_labels)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=min(32, len(new_data)), 
                shuffle=True
            )
            # Optimize the new class
            for epoch in range(epochs):
                epoch_loss = 0
                self.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    predictions = self(batch_X)
                    loss = torch.nn.functional.cross_entropy(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                print(f"Softmax FSL Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
            
    def evaluate_model(self, X, y, conf_matrix=False, sub_acc=False):
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