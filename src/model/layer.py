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

    def __init__(self, input_dim, num_prototypes, num_classes, lambda_val=1.0):
        super(GLVQ, self).__init__()
        # Number of classes in the dataset
        self.num_classes = num_classes
        self.lambda_val = lambda_val
        # Number of prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))
        # Prototype label
        self.prototype_labels = nn.Parameter(torch.tensor([i % num_classes for i in range(num_prototypes)]),
                                             requires_grad=False)
    
    def forward(self, x, y):
        # Distance between input data and prototypes
        distances = torch.cdist(x, self.prototypes)  # (batch_size, num_prototypes)
        
        positive_distances = torch.full((x.size(0),), float('inf')).to(x.device)
        negative_distances = torch.full((x.size(0),), float('inf')).to(x.device)
        
        same_class_mask = (self.prototype_labels.unsqueeze(0) == y.unsqueeze(1)).to(x.device)
        different_class_mask = ~same_class_mask

        positive_distances = torch.where(same_class_mask, distances, torch.full_like(distances, float('inf'))).min(dim=1).values
        negative_distances = torch.where(different_class_mask, distances, torch.full_like(distances, float('inf'))).min(dim=1).values

        mu = (positive_distances - negative_distances) / (positive_distances + negative_distances)
        loss = torch.mean(1 / (1 + torch.exp(-self.lambda_val * mu)))
        
        return loss
    
    def predict(self, x):
        """
        Predicts the class label for the input data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Predicted class labels.
        """

        distances = torch.cdist(x, self.prototypes)
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


class GLMVQ(nn.Module):
    def __init__(self, input_dim, num_prototypes, num_classes, lambda_val=1.0):
        super(GLMVQ, self).__init__()
        self.num_classes = num_classes
        self.lambda_val = lambda_val
        self.input_dim = input_dim
        
        # Prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))
        self.prototype_labels = nn.Parameter(
            torch.tensor([i % num_classes for i in range(num_prototypes)]),
            requires_grad=False
        )
        
        # Create a omega matrix with random normal values
        self.omega = nn.Parameter(
            torch.stack([
                torch.eye(input_dim) + torch.randn(input_dim, input_dim) * 0.01 for i in range(num_classes)
            ])
        )

    def compute_distance(self, x, prototype, omega):
        """
        Calculates ||Omega * x - Omega * w||^2
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) oder (input_dim,)
            prototype: Prototype tensor of shape (input_dim,)
            omega: Transformation matrix of shape (input_dim, input_dim)
        """

        # Make sure that x has the right shape
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        
        # Transform x and prototype
        transformed_x = torch.matmul(omega, x.T).T
        transformed_p = torch.matmul(omega, prototype)
        
        dist = torch.sum((transformed_x - transformed_p) ** 2, dim=-1)
        return dist

    def forward(self, x, y):
        batch_size = x.size(0)
        
        positive_distances = torch.full((batch_size,), float('inf')).to(x.device)
        negative_distances = torch.full((batch_size,), float('inf')).to(x.device)
        
        for i in range(batch_size):
            sample = x[i]
            label = y[i]
            
            distances = torch.zeros(len(self.prototypes)).to(x.device)
            
            for j, prototype in enumerate(self.prototypes):
                proto_label = self.prototype_labels[j]
                omega = self.omega[proto_label]
                distances[j] = self.compute_distance(sample.unsqueeze(0), prototype, omega)
            
            same_class_mask = self.prototype_labels == label
            diff_class_mask = ~same_class_mask
            
            if same_class_mask.any():
                positive_distances[i] = distances[same_class_mask].min()
            if diff_class_mask.any():
                negative_distances[i] = distances[diff_class_mask].min()
        
        mu = (positive_distances - negative_distances) / (positive_distances + negative_distances)
        loss = torch.mean(1 / (1 + torch.exp(-self.lambda_val * mu)))
        
        # Regularizaton term
        omega_reg = torch.norm(self.omega.view(-1))
        loss = loss + 0.01 * omega_reg
        
        return loss

    def predict(self, x):
        """
        Predicts the class label for the input data.
        """
        batch_size = x.size(0)
        all_distances = torch.zeros(batch_size, len(self.prototypes)).to(x.device)
        
        # Compute distances to all prototypes
        for i in range(batch_size):
            for j, prototype in enumerate(self.prototypes):
                proto_label = self.prototype_labels[j]
                omega = self.omega[proto_label]
                all_distances[i, j] = self.compute_distance(x[i], prototype, omega)
        
        # Get closest prototype indice
        closest_prototype_idx = torch.argmin(all_distances, dim=1)
        predicted_labels = self.prototype_labels[closest_prototype_idx]
        
        return predicted_labels

    def add_prototypes(self, new_prototypes, new_labels):
        """
        Adds new prototypes for few-shot learning.
        """
        # Concat the new prototypes and labels with existing ones
        self.prototypes = nn.Parameter(torch.cat([self.prototypes, new_prototypes], dim=0))
        self.prototype_labels = nn.Parameter(torch.cat([self.prototype_labels, new_labels]), requires_grad=False)