import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .m1 import SSBDModel1
from .m2 import load_ssbd_model2, m2_identify

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
ID2ACTION = ["noclass", "armflapping", "headbanging", "spinning"]


"""
    Pipeline Code. Outputs the exact action name out of ID2ACTION
"""
def detect_actions(video_path):
    results = []
    m1 = SSBDModel1(**M1_PARAMS)
    m2 = load_ssbd_model2()
    
    video = prefetch_call(video_path) # TODO
    prob_action = F.sigmoid(m1(video))
    action_id = -1
    
    if prob_action > 0.5:
        action_id = np.argmax(m2_identify(video_path), axis = 1)
        
    return ID2ACTION[action_id + 1]

