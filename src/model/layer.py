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

    def __init__(self, input_dim, num_prototypes, num_classes, alpha=1.0):
        super(GLVQ, self).__init__()
        # Number of classes in the dataset
        self.num_classes = num_classes
        self.alpha = alpha
        # Number of prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))
        # Prototype label
        self.prototype_labels = nn.Parameter(torch.tensor([i % num_classes for i in range(num_prototypes)]),
                                             requires_grad=False)
    
    def forward(self, x, y):

        dist = self.compute_dist(x)

        d1, d2 = self.prototype_dist(dist, y)

        mu = self.mu(d1, d2)

        return self.compute_loss(mu)
    
    def prototype_dist(self, dist, y):
        correct_mask = torch.zeros_like(dist).bool()
        correct_mask[torch.arange(y.size(0)), y] = True

        # Mask for incorrect prototypes
        incorrect_mask = ~correct_mask

        # Extract distances for correct and incorrect prototypes
        d1 = torch.min(dist.masked_fill(~correct_mask, float('inf')), dim=1).values
        d2 = torch.min(dist.masked_fill(~incorrect_mask, float('inf')), dim=1).values

        return d1, d2
    
    def compute_dist(self, x):
        x_trans = x.unsqueeze(1)

        prototype_trans = self.prototypes.unsqueeze(0)

        distances = torch.sum((x_trans - prototype_trans)**2, dim=2)
        return distances
    
    def mu(self, d1, d2):
        return (d1-d2)/(d1+d2)
    
    def compute_loss(self, mu):
        f_mu = torch.sigmoid(-self.alpha * mu)
        return torch.mean(f_mu)


    
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

class GMLVQ(GLVQ):
    def __init__(self, input_dim, num_prototypes, num_classes, alpha=1.0):
        super(GMLVQ, self).__init__(input_dim, num_prototypes, num_classes, alpha)
        # Introduce a learnable Lambda matrix (Omega.T @ Omega)
        self.omega = nn.Parameter(torch.randn(input_dim, input_dim))

    def compute_dist(self, x):
        """
        Overrides the distance computation to use the quadratic form:
        d_j = (x - w_j)^T * Lambda * (x - w_j),
        where Lambda = Omega.T @ Omega.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Distances to all prototypes in the transformed space.
        """
        # Compute Lambda = Omega.T @ Omega (positive semi-definite matrix)
        lamb = self.omega.T @ self.omega

        # Compute (x - w_j) for all prototypes
        diff = x.unsqueeze(1) - self.prototypes.unsqueeze(0)  # (batch_size, num_prototypes, input_dim)

        # Compute quadratic form for distances
        dist = torch.einsum('bni,ij,bnj->bn', diff, lamb, diff)  # (batch_size, num_prototypes)
        return dist