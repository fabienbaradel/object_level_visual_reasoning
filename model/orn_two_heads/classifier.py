from __future__ import print_function
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, size_input, size_output):
        super(Classifier, self).__init__()
        # Basic settings
        self.size_input = size_input
        self.size_output= size_output

        # FC layer
        self.fc = nn.Linear(self.size_input, self.size_output)

    def forward(self, x):

        preds = self.fc(x)

        return preds