import torch.nn as nn
import torch

"""
    @class (2+1)D CNN used in detecting the presence of actions
        @param in_channels: The number of input channels to the model
        @param intermediate: The number of intermediate channels (i.e size of the input to the second block of the model)
        @param out_channels: The number of the output channels of the model
        @param kernel_size: Array containing the respective dimensions of the kernels
        @param strides: Array containing the respective convolutional stride values
"""
class TwoPlusOneD_CNN(nn.Module):
    def __init__(self, in_channels, intermediate, out_channels, kernel_size, strides):
        super(TwoPlusOneD_CNN, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, intermediate, kernel_size = (1, kernel_size[1], kernel_size[2]), stride = (1, strides[1], strides[2]), bias = True, padding = 'valid'),
            nn.BatchNorm3d(intermediate),
            nn.ReLU()
        )
    
        self.temp_block = nn.Sequential(
            nn.Conv3d(intermediate, out_channels, kernel_size = (kernel_size[0], 1, 1), stride = (strides[0], 1, 1), bias = True, padding = 'valid'),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.temp_block(x)
        return x

"""
    @class Fully connected NN used in detecting the presence of actions
        @param input_size: The dimension of the model's input
        @param size_1: The dimension of the first hidden layer
        @param size_2: The dimension of the second hidden layer
            @note forward() method can be implemented by the user to use this variable
        @param size_3: The dimension of the third hidden layer
            @note forward() method can be implemented by the user to use this variable
        @param n_classses: Array containing the respective convolutional stride values
"""
class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, size_1, size_2, size_3, n_classes):
        super(FullyConnectedNet, self).__init__()
        self.fcn = nn.Sequential(
            nn.Linear(input_size, size_1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(size_1),
            
            nn.Linear(size_1, n_classes)
        )
        
    def forward(self, x):
        return self.fcn(x)

"""
    @class M1: Used to detect the presence or actions of stimming behaviour.
        @component (2+1)D CNN
        @component Fully-connected NN
"""
class SSBDModel1(nn.Module):
    def __init__(self, in_channels, intermediate, 
                 out_channels, kernel_size, strides, pooling_size, 
                 pooling_strides, size_1, size_2, size_3):
        super(SSBDModel1, self).__init__()
        
        self.net = nn.Sequential(
            TwoPlusOneD_CNN(in_channels, intermediate,  
                            out_channels, kernel_size, strides),
            nn.AvgPool3d(kernel_size = pooling_size, stride = pooling_strides),
        )
        
        self.fcn = nn.Sequential(
            FullyConnectedNet(288, size_1, size_2, size_3, 1)
        )
        
    def forward(self, x):
        x = self.conv_end(x)
        x = self.globalAvgPool(x)
        x = x.view(x.size(0), -1)
        return self.fcn(x)