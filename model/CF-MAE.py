# Imports
import torch
import torch.nn as nn

import timm

import numpy as np




class TemporalGeoMAE(nn.Module):

    def __init__(self,
                 model_name
                 ):
        super().__init__()


        self.encoder = timm.create_model(model_name=model_name,pretrained=False,drop_rate=0.2,num_classes = 0,global_pool= '')

