#------------------------------------------------------------
# This file contains the models in consideration
#------------------------------------------------------------
import torch
import torch.nn as nn

class VanillaNN(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim, if_dropout=False, dropout_rate=0.5, if_batchnorm=False):
        '''
        Parameters:
        input_dim: int
            The input dimension of the neural network
        hidden_units: list
            The number of hidden units in each hidden layer
        output_dim: int
            The output dimension of the neural network
        if_dropout: bool
            Whether to include dropout layers
        dropout_rate: float
            The dropout rate
        if_batchnorm: bool
            Whether to include batch normalization layers
        '''
        super(VanillaNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.if_dropout = if_dropout
        self.dropout_rate = dropout_rate
        self.if_batchnorm = if_batchnorm
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(self.input_dim, self.hidden_units[0]))
        
        # Hidden layers
        for i in range(1, len(self.hidden_units)):
            if self.if_batchnorm:
                self.layers.append(nn.BatchNorm1d(self.hidden_units[i-1]))
            if self.if_dropout:
                self.layers.append(nn.Dropout(self.dropout_rate))
            self.layers.append(nn.Linear(self.hidden_units[i-1], self.hidden_units[i]))
        
        # Output layer
        if self.if_batchnorm:
            self.layers.append(nn.BatchNorm1d(self.hidden_units[-1]))
        if self.if_dropout:
            self.layers.append(nn.Dropout(self.dropout_rate))
        self.layers.append(nn.Linear(self.hidden_units[-1], self.output_dim))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = torch.relu(x)
        x = self.layers[-1](x)
        return x
    