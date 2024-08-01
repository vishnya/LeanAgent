# Simple program to generate some fake data and then run on the first GPU available using PyTorch to
# raise the GPU utilization

import numpy as np
import torch

# Generate some fake data
data = np.random.randn(1000, 1000).astype(np.float32)

# Move the data to the GPU
data = torch.tensor(data).cuda()

# Perform some operations on the data
for i in range(1000000000):
    data = data @ data
