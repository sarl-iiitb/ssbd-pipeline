import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F

"""
    @class Prefetch used to classify YOLO bounding boxes as adults or children
           This model was used as a part of preprocessing.
        @param dropout_rate: The dropout rate used by the model
            @default 0.40
        @param dim_fc_layer_1: The dimension of the first hidden layer
            @default 256
        @param dim_fc_layer_2: The dimension of the second hidden layer
            @default 64
"""
class PreFetchNet(nn.Module):
    def __init__(self, dropout_rate = 0.40, dim_fc_layer_1 = 256, dim_fc_layer_2 = 64):
        super(PreFetchNet, self).__init__()

        self.baseline = models.vgg19_bn(weights='IMAGENET1K_V1')
        
        for param in self.baseline.parameters():
            param.requires_grad = False
        for param in self.baseline.classifier.parameters():
            param.requires_grad = False
            
        self.dropout_rate = dropout_rate
        self.dim_fc_layer_1 = dim_fc_layer_1
        self.dim_fc_layer_2 = dim_fc_layer_2

        # FC Layer
        self.fc1 = nn.Linear(1000, self.dim_fc_layer_1)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(self.dim_fc_layer_1)
        self.fc2 = nn.Linear(self.dim_fc_layer_1, self.dim_fc_layer_2)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.batchnorm2 = nn.BatchNorm1d(self.dim_fc_layer_2)
        self.fc3 = nn.Linear(self.dim_fc_layer_2, 2)

    def forward(self, x):
        # Get the R^1000 weight vector from the base model
        x = self.baseline(x)

        # Fully-connected classifier network
        x = F.sigmoid(self.fc1(x))
        x = self.dropout1(x)
        x = self.batchnorm1(x)
        x = F.sigmoid(self.fc2(x))
        x = self.dropout2(x)
        x = self.batchnorm2(x)
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(x)
        return x