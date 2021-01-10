# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 12:15:14 2021

@author: jun_h
"""

# import libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from autograd import Variable

# create the architecture of the Neural Network

class Network(nn.Module):
    
    # this can be arbitrarily chosen, 30 is most optimized size after multiple trials
    hidden_size = 30
    
    def __init__(self, input_size, output_size):
        # initialize object using nn.Module constructor
        super(Network, self).__init__()
        
        # initialize the the input dimension size, and the output dimension size
        self.input_size = input_size
        self.output_size = output_size
        
        # connect the neurons from the inner layer to the hidden layer, "fc" means "Full Connection (Linear)" : all neurons of the input layer is connected to the hidden layer.
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # connect the neurons from the hidden layer to the output layer
        self.fc2 = nn.Linear(hidden_size, output_size)