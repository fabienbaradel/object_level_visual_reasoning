import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import ipdb

class AggregationRelations(nn.Module):
    def __init__(self):
        super(AggregationRelations, self).__init__()

    def forward(self, relational_reasoning_vector):
        # Basic summation
        B, T, K2, _ = relational_reasoning_vector.size()

        output = torch.sum(relational_reasoning_vector, 2)


        return output  # (B,T-1,512)