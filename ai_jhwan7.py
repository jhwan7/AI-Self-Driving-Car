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
        
    def push(self, event): # event is defined by 4 elements lastState, newState, lastAction, lastReward
         self.memory.append(event)
         
         # only store "capacity" amounts of event. Remove if it surpasses.
         if len(self.memory) > self.capacity:
             del self.memory[0]
             
    # the batch of data needs to be selected with caution.
    def sample (self, batch_size): 
        # zip (*): reshapes the list to a new form. Needed to put it into torch variables
        samples = zip(*random.sample(self.memory, batch_size))
        
        # map samples to torch variables
        # arg1: function to transform samples to torch variables, arg2: the sample
        # Variable(): converts samples to torch variable.
        # for each batch we need to concactenate it to the first dimension. Ensures everything is well alligned (states, action, and rewards responds to the same time, T)
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
    
# Implement Deep Q Learning

class Dqn():
    
    # take in all args needed for the Network, ReplayMemory class + the gamma parameter
    def __init__(self, input_size, output_size, gamma):
        self.gamma = gamma
        
        # reward window, mean of the last 100 rewards
        self.reward_window = []
        
        # create the actual model
        self.model = Network(input_size, output_size)
        
        # create the memory
        self.memory = ReplayMemory(100000)
        
        # optimizer, pass model parameters and learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr =0.001 )
        
        # Network expectes data to come in as a batch in the first dimension. We introduce a fake dimension which corresponds to the batch
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        
        # index representing which action it took
        self.last_action = 0;
        
        # float number that represents the reward
        self.last_reward = 0;
        
        
    def select_action (self, input_state):
        # generate the Q values required in Softmax by using our model that generates Q values.
        probs = F.softmax(self.model(Variable(input_state, volatile = True)) * 7) # Temperature (T) = How sure the AI agent is about an action, Bigger T increases the certainty of which action to take
        
        # multinomial returns a random value depending on the property
        action = probs.multinomial()
        
        # action returns with fake batch, so we need to get data at index 0,0 where our action is stored
        return action.data[0,0]
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        