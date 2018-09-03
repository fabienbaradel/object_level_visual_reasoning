import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import ipdb


class EncoderMLP(nn.Module):
    def __init__(self, input_size=149, list_hidden_size=[100, 100], relu_activation=True, p_dropout=0.5):
        super(EncoderMLP, self).__init__()
        self.input_size = input_size
        self.list_hidden_size = list_hidden_size

        # Encoder
        self.encoder = nn.Sequential()
        current_input_size = self.input_size
        for i, hidden_size in enumerate(self.list_hidden_size):
            # Add the linear layer
            self.encoder.add_module('linear_{}'.format(i), nn.Linear(current_input_size, hidden_size))
            self.encoder.add_module('dropout_{}'.format(i), nn.Dropout(p=p_dropout))
            current_input_size = hidden_size
            # Add ReLu activation
            self.encoder.add_module('relu_{}'.format(i), nn.ReLU())

    def forward(self, x):
        size_x = x.size()

        # Transform to get the corresponding input vector size
        if size_x[-1] == self.input_size:
            x_input = x
        elif size_x[-1] * size_x[-2] == self.input_size:
            if len(size_x) == 5:
                B, T, K, W, H = size_x
            elif len(size_x) == 4:
                B, T, K, W = size_x
                H = 1
            x = x.contiguous()
            x_input = x.view(B, T, K, W * H)
        else:
            raise Exception

        # Encoder
        z = self.encoder(x_input)

        return z