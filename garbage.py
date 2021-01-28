import torch
import torch.nn as nn


a = nn.Conv2d(2, 2, 3, 1, 0, groups=2)
print(a.weight)
print(a.weight.shape)
print(a.bias)