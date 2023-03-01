"""
A way to experiment with the Q Learning model
"""
import torch.nn as nn

class LinearQNetwork(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        # let us create two linear layers
        self.linear_one = nn.Linear(input_size, hidden_size)
        self.linear_two = nn.Linear(hidden_size, output_size)
    
    