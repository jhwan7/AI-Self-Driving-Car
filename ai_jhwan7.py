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
    
    def __init__(self, input_size, output_size):
        # initialize object using nn.Module constructor
        super(Network, self).__init__()
        
        # initialize the the input dimension size, and the output dimension size
        self.input_size = input_size
        self.output_size = output_size
        
        # this can be arbitrarily chosen, 30 is most optimized size after multiple trials
        self.hidden_size = 30
        
        # connect the neurons from the inner layer to the hidden layer, "fc" means "Full Connection (Linear)" : all neurons of the input layer is connected to the hidden layer.
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        
        # connect the neurons from the hidden layer to the output layer
        self.fc2 = nn.Linear(self.hidden_size, output_size)  
        
    # determine and return the q-values after it is processed with the AI using the input "state"
    def forward(self, state):
        # activate the hidden neurons, by applying the rectifier (relu) function
        # get hidden neurons by passing the inputState through the fc1 (Passing input values to the hidden neurons)
        hidden_layer_neuron = F.relu(self.fc1(state))
        
        #  From the full connection between the hidden and output layer, pass on the hidden layer input to get the resulting output neurons.
        # This will not be directly interpreted as the resulting action yet. It will go through a Softmax function to finalize the action
        q_values = self.fc2(hidden_layer_neuron)
        return q_values 
    
# implement Experience Replay 
# instead of updating every iteration when the state changes, we hold experience in memory. Enhance the machine learning in long term correlation
class ReplayMemory(object):
    
    def __init__(self, capacity = 100):
        
        # defines the length of the array that represents the memory
        self.capacity = capacity
        
        # stores the last "capacity" amount of states
        self.memory = []
        