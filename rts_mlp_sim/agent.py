import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    A simple Multilayer Perceptron model.
    It will take the game state as input and output the Q-values for each possible action.
    """
    def __init__(self, input_size, output_size):
        """
        Initializes the layers of the network.
        :param input_size: The size of the state vector.
        :param output_size: The number of possible actions.
        """
        super(MLP, self).__init__()
        hidden_size = 16  # Reduced for better visualization
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        :param x: The input tensor (state).
        :return: The output tensor (Q-values for actions).
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
