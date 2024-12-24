import torch
import torch.nn as nn
import torch.nn.functional as F


class GLVQ(nn.Module):
    
    """
    Generalized Learning Vector Quantization (GLVQ) Layer for PyTorch.
     
    Args:
        input_dim (int): Dimensionality of the input data.
        num_prototypes (int): Number of prototypes.
        num_classes (int): Number of classes in the dataset.

    Attributes:
        prototypes (nn.Parameter): Trainable parameter for the prototypes.
        prototype_labels (nn.Parameter): Fixed parameter for the prototype labels.
    
    Returns:
        tuple(torch.Tensor, torch.Tensor): Tuple of positive and negative distances.

    """

    def __init__(self, input_dim, num_prototypes_per_class, num_classes, alpha=1.0):
        super(GLVQ, self).__init__()
        # Number of classes in the dataset
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_classes = num_classes
        self.alpha = alpha
        self.num_prototypes = num_prototypes_per_class * num_classes
        # Number of prototypes
        self.prototypes = nn.Parameter(torch.empty(self.num_prototypes, input_dim))

        # Prototype label
        self.prototype_labels = nn.Parameter(torch.tensor([i % num_classes for i in range(self.num_prototypes)]),
                                             requires_grad=False)
        
        self.initialized = False
    
    def forward(self, x, y):
        if not self.initialized:
            self.initialize_prototypes(x, y)
        dist = self.compute_dist(x)
        d1, d2 = self.prototype_dist(dist, y)
        mu = self.mu(d1, d2)
        loss = self.compute_loss(mu)

        with torch.no_grad():
            predictions = self.predict(x)
            accuracy = (predictions == y).float().mean()
            print(f"Batch Accuracy: {accuracy.item():.4f}")
            print(f"d1 mean: {d1.mean().item():.4f}, d2 mean: {d2.mean().item():.4f}")
        return loss
    
    def prototype_dist(self, dist, y):
        correct_mask = self.prototype_labels.unsqueeze(0) == y.unsqueeze(1)
        incorrect_mask = ~correct_mask

        d1 = torch.min(dist.masked_fill(~correct_mask, float('inf')), dim=1).values
        d2 = torch.min(dist.masked_fill(~incorrect_mask, float('inf')), dim=1).values

        return d1, d2
    
    def initialize_prototypes(self, X, y):
        """
        Initializes the prototypes based on the input data and uses a 
        k-means like approach to choose the prototypes.

        Args:
            X (torch.Tensor): Input data of the shape [N, D].
            y (torch.Tensor): Labels of the input data.
        """
        if self.initialized:
            return
        
        # Loop through each class
        for class_idx in range(self.num_classes):
            # Get datapoijnts for the current class
            class_mask = y == class_idx
            class_samples = X[class_mask]
            
            if len(class_samples) == 0:
                continue

            num_proto = self.num_prototypes_per_class

            # Choose random prototypes from the class samples
            proto_indices = torch.randperm(len(class_samples))[:num_proto]
            class_protos = class_samples[proto_indices]
            
            # Set the prototypes
            start_idx = class_idx * self.num_prototypes_per_class
            end_idx = start_idx + self.num_prototypes_per_class
            self.prototypes.data[start_idx:end_idx] = class_protos
        
        self.initialized = True


    
    def compute_dist(self, x):
        """
        x_trans = x.unsqueeze(1)
        prototype_trans = self.prototypes.unsqueeze(0)
        distances = torch.sum((x_trans - prototype_trans)**2, dim=2)
        """
        diff = x.unsqueeze(1) - self.prototypes.unsqueeze(0)
        return torch.sum(diff ** 2, dim=2)
    
    def mu(self, d1, d2):
        epsilon = 1e-9
        return (d1-d2)/(d1+d2 + epsilon)
    
    def compute_loss(self, mu):
        f_mu = 1 / 1 + torch.sigmoid(self.alpha * mu)
        return torch.mean(f_mu)


    
    def predict(self, x):
        """
        Predicts the class label for the input data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Predicted class labels.
        """

        distances = self.compute_dist(x)
        predicted_labels = self.prototype_labels[torch.argmin(distances, dim=1)]
        return predicted_labels

    
    def add_prototypes(self, new_prototypes, new_labels):
        """
        Adds new prototypes for few-shot learning.

        Args:
            new_prototypes (torch.Tensor): New prototypes to add.
            new_labels (torch.Tensor): Labels for the new prototypes.
        """
        # Concatenate the new prototypes and labels with existing ones
        self.prototypes = nn.Parameter(torch.cat([self.prototypes, new_prototypes], dim=0))
        self.prototype_labels = nn.Parameter(torch.cat([self.prototype_labels, new_labels]), requires_grad=False)

class GMLVQ(GLVQ):
    def __init__(self, input_dim, num_prototypes, num_classes, alpha=1.0):
        super(GMLVQ, self).__init__(input_dim, num_prototypes, num_classes, alpha)
        # Learnable feature matrix (Omega.T @ Omega)
        self.omega = nn.Parameter(torch.eye(input_dim))

    def compute_dist(self, x):
            """
            Overrides the distance computation to include the learnable metric transformation.

            Args:
                x (torch.Tensor): Input data.

            Returns:
                torch.Tensor: Transformed distances between input and prototypes.
            """
            # Apply metric transformation
            x_transformed = torch.matmul(x, self.omega.T)
            prototypes_transformed = torch.matmul(self.prototypes, self.omega.T)

            # Compute squared Euclidean distance in transformed space
            diff = x_transformed.unsqueeze(1) - prototypes_transformed.unsqueeze(0)
            return torch.sum(diff ** 2, dim=2)