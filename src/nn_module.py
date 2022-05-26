import torch
from torch import nn


class FirstNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

first = FirstNN()
input = torch.tensor(1.0)
out = first(input)
print(out)
