import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):
    """
    Basic implemantation of a fully connected autoencoder
    """
    def __init__(self, input_size, bottleneck_size):
        super(AE, self).__init__()

        self.input_size = input_size
        self.bottleneck_size = bottleneck_size

        # Encoder layers
        self.linear1 = nn.Linear(self.input_size, self.input_size // 2)
        self.linear2 = nn.Linear(self.input_size // 2, self.bottleneck_size)

        # Decoder layers

        self.linear_1 = nn.Linear(self.bottleneck_size, self.input_size // 2)
        self.linear_2 = nn.Linear(self.input_size // 2, self.input_size)

    def encode(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x
    
    def decode(self, x):
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


