# This is a neural net class inherits pytorch nn class atributes and functions

import torch.nn.functional as F
import torch.nn as nn


class NeuralNet(nn.Module): # inherit from nn
    def __init__(self):
        
        super().__init__() # run nn.Module __init__

        self.fully_connected_layer1 = nn.Linear(28*28, 128) # 28 represents dimension of the MNIST data set 
        self.fully_connected_layer2 = nn.Linear(128, 128)   # and 128 is the number of neurons
        self.fully_connected_layer3 = nn.Linear(128, 128)
        self.fully_connected_layer4 = nn.Linear(128, 128)
        self.fully_connected_layer5 = nn.Linear(128, 128)
        self.fully_connected_layer6 = nn.Linear(128, 10)

        # Use rectified linear for firing the neurons
    def forward(self, input):
        input = F.relu(self.fully_connected_layer1(input))
        input = F.relu(self.fully_connected_layer2(input))
        input = F.relu(self.fully_connected_layer3(input))
        input = F.relu(self.fully_connected_layer4(input))
        input = F.relu(self.fully_connected_layer5(input))
        input = self.fully_connected_layer6(input)

        return F.log_softmax(input, dim=1)




