import torch
import torch.nn as nn
import torch.nn.functional as F

from m1 import SSBDModel1
from m2 import SSBDModel2

import torchvision.models as models

"""
    Arguments for the M1 model
"""
M1_PARAMS = dict(
    in_channels = 3, 
    intermediate = 16, 
    out_channels = 8, 
    kernel_size = [3, 3, 3], 
    strides = [1, 1, 1], 
    pooling_size = [1, 3, 3], 
    pooling_strides = [1, 3, 3], 
    size_1 = 128, 
    size_2 = 64, 
    size_3 = 16
)

"""
    Arguments for the M2 model
"""
M2_PARAMS = dict(
    dropout_rate = 0.50,
    dim_embedding = 64,
    dim_hidden = 64,
    num_lstm_layers = 32,
    dim_fc_layer_1 = 128,
    dim_fc_layer_2 = 8,
    base_model = models.resnet18(pretrained = True),
    n_frames = 40,
    use_movenet = True,
    n_classes = 3
)

"""
    Pipelined model for self-stimulatory behaviour detection
        @note Expected output of the model
            Noclass: 0
            Armflapping: 1
            Headbanging: 2
            Spinning: 3
"""
class SSBDPipeline(nn.Module):
    def __init__(self):
        self.m1 = SSBDModel1(**M1_PARAMS)
        self.m2 = SSBDModel2(**M2_PARAMS)

    def forward(self, x):
        vid, movenet_x = x
        prob_action = F.sigmoid(self.m1(vid))

        if prob_action > 0.5:
            _, pred = torch.max(F.softmax(self.m2(x)).data, 1)
            return int(float(pred.cpu().numpy()[0])) + 1

        return 0