import torch
from torch.autograd import Variable
import numpy as np

a = np.zeros((2, 3))
print("np:", a)
print("np.shape:", a.shape)
print("np.size:", a.size)
a = torch.tensor(a)
print("tensor:", a)
print("tensor.shape:", a.shape)
print("tensor.size:", a.size)
a = Variable(a)
print("va:", a)
print("va.shape:", a.shape)
print("va.size:", a.size)