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
