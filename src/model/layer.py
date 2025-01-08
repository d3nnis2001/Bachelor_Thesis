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
        self.prototype_labels = nn.Parameter(torch.tensor([i // num_prototypes_per_class for i in range(self.num_prototypes)]), requires_grad=False)
        
        # Variable so initialization is only done once
        self.initialized = False
    
    def forward(self, x, y, t_value):
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
        loss = self.compute_loss(mu, t_value)
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
        d1 = torch.min(dist.masked_fill(~correct_mask, float("inf")), dim=1).values
        d2 = torch.min(dist.masked_fill(~incorrect_mask, float("inf")), dim=1).values

        return d1, d2
    
    def initialize_prototypes(self, X, y):
        # TODO: Maybe implement a small shift of mean
        """
        Initializes the prototypes based on the class mean.

        Parameters:
        -----------
        X : torch.Tensor
            Input data
        y : torch.Tensor
            Target labels
        """
        X = X.to(self.prototypes.device)
        y = y.to(self.prototype_labels.device)
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
    
    def compute_loss(self, mu, t_value=1):
        """
        Computes the GLVQ loss using the sigmoid function.

        Parameters:
        -----------
        mu : torch.Tensor
            Difference between the distances to the correct and incorrect prototypes.
        """
        # Alpha scaling with epochs
        alpha =  torch.log(1 + t_value) / self.alpha
        f_mu = torch.sigmoid(alpha * mu)
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
        self.num_classes += 1
        self.num_prototypes += self.num_prototypes_per_class
        new_prototypes = new_prototypes.to(self.prototypes.device)
        new_labels = new_labels.to(self.prototype_labels.device)
        # Concatenate the new prototypes and labels with existing ones
        self.prototypes = nn.Parameter(torch.cat([self.prototypes, new_prototypes], dim=0))
        self.prototype_labels = nn.Parameter(torch.cat([self.prototype_labels, new_labels]), requires_grad=False)

    # Getter and Setter
    def get_prototypes(self):
        return self.prototypes

    def get_prototype_labels(self):
        return self.prototype_labels
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_num_prototypes(self):
        return self.num_prototypes


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

        # Learnable feature matrix
        self.omega = nn.Parameter(torch.eye(input_dim))


    def compute_dist(self, x):
        """
        Compute distances using adaptive squared euclidian metric d_Λ(x,w) = (x-w)^T Λ (x-w).

        Parameters:
        -----------
        x : torch.Tensor
            Input data.
        """
        # Apply metric transformation
        x_transformed = torch.mm(x, self.omega.T)
        prototypes_transformed = torch.mm(self.prototypes, self.omega.T)

        # Compute squared Euclidean distance in transformed space
        diff = x_transformed.unsqueeze(1) - prototypes_transformed.unsqueeze(0)
        return torch.sum(diff ** 2, dim=2)
    
    def normalize_metric(self):
        """
        Normalize metric tensor to maintain Tr(Λ) = 1 as per https://www.cs.rug.nl/~biehl/Preprints/gmlvq.pdf
        """
        with torch.no_grad():
            lambda_matrix = torch.mm(self.omega.T, self.omega)
            trace = torch.trace(lambda_matrix)
            self.omega.data = self.omega.data / torch.sqrt(trace + 1e-10)

    def forward(self, x, y, t_value):
        """
        Forward pass with metric normalization

        Parameters:
        -----------
        x : torch.Tensor
            Input data
        y : torch.Tensor
            Target labels
        t_value : int
            Time value for loss function.
        """
        loss = super().forward(x, y, t_value)
        # Normalize metric
        self.normalize_metric()

        return loss