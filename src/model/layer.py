import torch
import torch.nn as nn
import torch.nn.functional as F


class GLVQ(nn.Module):
    
    """
    Generalized Learning Vector Quantization (GLVQ) Layer for PyTorch.
     
    Parameters:
    ----------
    input_dim : int
        Dimensionality of the input data.
    num_prototypes_per_class : int
        Number of prototypes per class.
    num_classes : int
        Number of classes in the dataset.
    alpha : float
        Alpha parameter for the sigmoid function.
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
        """
        Forward pass through the GLVQ layer.

        Parameters:
        -----------
        x : torch.Tensor
            Input data
        y : torch.Tensor
            Target labels
        """
        if not self.initialized:
            self.initialize_prototypes(x, y)
        dist = self.compute_dist(x)
        d1, d2 = self.prototype_dist(dist, y)
        mu = self.mu(d1, d2)
        loss = self.compute_loss(mu)
        return loss
    
    def prototype_dist(self, dist, y):
        """
        Calculates the distances to the correct and incorrect prototypes.
        
        Parameters:
        -----------
        dist : torch.Tensor
            Distance matrix between input and prototypes.
        y : torch.Tensor
            Target labels.

        """
        # Mask for correct and incorrect prototypes
        correct_mask = self.prototype_labels.unsqueeze(0) == y.unsqueeze(1)
        incorrect_mask = ~correct_mask

        # Get the distance to the closest correct and incorrect prototype
        d1 = torch.min(dist.masked_fill(~correct_mask, float('inf')), dim=1).values
        d2 = torch.min(dist.masked_fill(~incorrect_mask, float('inf')), dim=1).values

        return d1, d2
    
    def initialize_prototypes(self, X, y):
        """
        Initializes the prototypes based on the class mean.

        Parameters:
        -----------
        X : torch.Tensor
            Input data
        y : torch.Tensor
            Target labels
        """
        # Avoids multiple initializations
        if self.initialized:
            return
        # Iterate over all classes
        for c in range(self.num_classes):
            # Mask for the current class
            class_mask = y == c
            class_samples = X[class_mask]
            # Check if class has samples
            if len(class_samples) == 0:
                continue

            # Calc mean of the class samples
            class_mean = class_samples.mean(dim=0)

            # Do it for each prototype per class
            class_protos = class_mean.repeat(self.num_prototypes_per_class, 1)
            # Initialize the prototypes
            start = c * self.num_prototypes_per_class
            end = start + self.num_prototypes_per_class
            self.prototypes.data[start:end] = class_protos

        self.initialized = True


    
    def compute_dist(self, x):
        """
        Computes the squared Euclidean distance between input and prototypes.

        Parameters:
        -----------
        x : torch.Tensor
            Input data.
    
        """
        diff = x.unsqueeze(1) - self.prototypes.unsqueeze(0)
        return torch.sum(diff ** 2, dim=2)
    
    def mu(self, d1, d2):
        """
        Calculates the difference between the distances to the correct and incorrect prototypes.

        Parameters:
        -----------
        d1 : torch.Tensor
            Distance to the correct prototype.
        d2 : torch.Tensor
            Distance to the incorrect prototype.
        """
        # Small epsilon to avoid division by zero
        epsilon = 1e-9
        return (d1-d2)/(d1+d2 + epsilon)
    
    def compute_loss(self, mu):
        """
        Computes the GLVQ loss using the sigmoid function.

        Parameters:
        -----------
        mu : torch.Tensor
            Difference between the distances to the correct and incorrect prototypes.
        """
        f_mu = torch.sigmoid(self.alpha * mu)
        return torch.mean(f_mu)


    
    def predict(self, x):
        """
        Predicts the class label for the input data.

        Parameters:
        -----------
        x : torch.Tensor
            Input data.
        """

        distances = self.compute_dist(x)
        # Predict the class label based on the closest prototype for each input
        predicted_labels = self.prototype_labels[torch.argmin(distances, dim=1)]
        return predicted_labels

    
    def add_prototypes(self, new_prototypes, new_labels):
        """
        Adds new prototypes for few-shot learning.

        Parameters:
        -----------
        new_prototypes : torch.Tensor
            New prototypes.
        new_labels : torch.Tensor
            Labels for the new prototypes.
        """
        # Concatenate the new prototypes and labels with existing ones
        self.prototypes = nn.Parameter(torch.cat([self.prototypes, new_prototypes], dim=0))
        self.prototype_labels = nn.Parameter(torch.cat([self.prototype_labels, new_labels]), requires_grad=False)

class GMLVQ(GLVQ):
    """
    Generalized Matrix Learning Vector Quantization (GMLVQ) Layer for PyTorch.
    
    Parameters:
    -----------
    input_dim : int
        Dimensionality of the input data.
    num_prototypes : int
        Number of prototypes.
    num_classes : int
        Number of classes in the dataset.
    alpha : float
        Alpha parameter for the sigmoid function.
    """
    def __init__(self, input_dim, num_prototypes, num_classes, alpha=1.0):
        super(GMLVQ, self).__init__(input_dim, num_prototypes, num_classes, alpha)
        # Learnable feature matrix (Omega.T @ Omega)
        self.omega = nn.Parameter(torch.eye(input_dim))

    def compute_dist(self, x):
            """
            Overrides the distance computation to include the learnable metric transformation.

            Parameters:
            -----------
            x : torch.Tensor
                Input data.
            """
            # Apply metric transformation
            x_transformed = torch.matmul(x, self.omega.T)
            prototypes_transformed = torch.matmul(self.prototypes, self.omega.T)

            # Compute squared Euclidean distance in transformed space
            diff = x_transformed.unsqueeze(1) - prototypes_transformed.unsqueeze(0)
            return torch.sum(diff ** 2, dim=2)