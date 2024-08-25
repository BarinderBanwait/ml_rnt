#------------------------------------------------------------
# This file contains the models in consideration
#------------------------------------------------------------

# Importing the required libraries
import torch
import torch.nn as nn

# A vanilla neural network model with n hidden layers in pytorch
# The hidden layer units are customizable
class VanillaNN(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim):
        '''
        Parameters:
        input_dim: int
            The input dimension of the neural network
        hidden_units: list
            The number of hidden units in each hidden layer
        output_dim: int
            The output dimension of the neural network
        '''
        super(VanillaNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.hidden_units[0]))
        for i in range(1, len(self.hidden_units)):
            self.layers.append(nn.Linear(self.hidden_units[i-1], self.hidden_units[i]))
        self.layers.append(nn.Linear(self.hidden_units[-1], self.output_dim))
        
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = torch.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x
    