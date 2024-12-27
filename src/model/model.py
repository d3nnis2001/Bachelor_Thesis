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
import numpy as np

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
    def __init__(self, version="GLVQ", num_prototypes_per_class=1, num_classes=8, epochs=5, optimizer_type="ADAM", learning_rate=0.001, batch_size=128, device=None):
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
        
        # Dense Feature Extraction
        self.dense_features = nn.Sequential(
            nn.Linear(64 * 10 * 64, 300),
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
        """
        x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1) 
        features = self.dense_features(x)
        return features
    
    def forward(self, x, y=None):
        """
        Forward pass through the network
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
    
    def fit(self, X, y):
        """
        Fits the model to the input train data X and y
        """
        # Trainingsmode
        self.train()
        X, y = X.to(self.device), y.to(self.device)

        # Load the data into the Tensorloader
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # Check which optimizer to use
        optimizer = (optim.Adam if self.optimizer_type == "ADAM" else optim.SGD)(
            self.parameters(), lr=self.learning_rate
        )
        
        # Keep track of the training history
        history = {
            'loss': [],
            'epoch': []
        }

        for epoch in range(self.epochs):
            epoch_losses = []
            # Go through the batches
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                if self.version in ["GLVQ", "GMLVQ"]:
                    loss = self(batch_X, batch_y)
                    print(f"Batch Loss (GLVQ/GMLVQ): {loss.item():.4f}")
                else:  # Softmax version
                    outputs = self(batch_X)
                    loss = F.nll_loss(outputs, batch_y)

                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            history['loss'].append(avg_loss)
            history['epoch'].append(epoch + 1)
            
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}')

        return history
                    
    def predict(self, X):
        """
        Predicts the class labels for the input data
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
        y_true = y.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        
        # Calculate accuracy
        accuracy = (predicted == y).float().mean()
        print(f'Test Accuracy: {accuracy.item():.4f}')
        
        # Define class names
        class_names = ['Flexion', 'Extension', 'Supination', 'Pronation', 
                    'Open', 'Pinch', 'Lateral pinch', 'Fist']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if conf_matrix:
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names)
            plt.title(f'Confusion Matrix - {self.version}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'confusion_matrix_{self.version}_{timestamp}.png')
            plt.show()
        
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        if sub_acc:
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=class_names))
            
            print("\nPer-class Accuracy:")
            for name, acc in zip(class_names, per_class_acc):
                print(f"{name}: {acc:.4f}")
        
        return {
            'accuracy': accuracy.item(),
            'confusion_matrix': cm,
            'per_class_accuracy': per_class_acc,
            'predictions': y_pred,
            'true_labels': y_true
        }