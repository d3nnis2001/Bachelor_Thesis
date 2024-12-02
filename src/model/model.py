import torch
import torch.nn as nn
from model.layer import GLVQ, GMLVQ
import torch.nn.functional as F

class CNet2D(nn.Module):
    def __init__(self, version="GLVQ", num_prototypes=8, num_classes=8):
        super(CNet2D, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 13), padding=(1, 6))
        self.bn1 = nn.BatchNorm2d(32)
        self.rrelu1 = nn.RReLU()
        self.drop1 = nn.Dropout(0.3)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 48, kernel_size=(3, 9), padding=(1, 4))
        self.bn2 = nn.BatchNorm2d(48)
        self.rrelu2 = nn.RReLU()
        self.drop2 = nn.Dropout(0.3)
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(48, 64, kernel_size=(3, 5), padding=(1, 2))
        self.bn3 = nn.BatchNorm2d(64)
        self.rrelu3 = nn.RReLU()
        self.drop3 = nn.Dropout(0.3)
        
        # Dense Block 1
        self.fc1 = nn.Linear(64 * 10 * 64, 300)
        self.fc_bn1 = nn.BatchNorm1d(300)
        self.fc_rrelu1 = nn.RReLU()
        self.fc_drop1 = nn.Dropout(0.3)
        
        # Dense Block 2 
        self.fc2 = nn.Linear(300, 10)
        self.fc_bn2 = nn.BatchNorm1d(10)
        self.fc_rrelu2 = nn.RReLU()
        if version == "Softmax":
            # Softmax Layer for classification
            self.fc3 = nn.Linear(10, 8)
        elif version == "GLVQ":
            self.fc3 = GLVQ(10, num_prototypes, num_classes)
        else:
            self.fc3 = GMLVQ(10, num_prototypes, num_classes)
        
    def forward(self, x, y=None):
        x = self.rrelu1(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (1, 2))
        x = self.drop1(x)
        
        x = self.rrelu2(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (1, 2))
        x = self.drop2(x)
        
        x = self.rrelu3(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, (1, 2))
        x = self.drop3(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc_rrelu1(self.fc_bn1(self.fc1(x)))
        x = self.fc_drop1(x)
        x = self.fc_rrelu2(self.fc_bn2(self.fc2(x)))

        return x if self.version in ["GLVQ", "GMLVQ"] else F.log_softmax(self.fc3(x), dim=1)