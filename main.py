import numpy as np
import pandas as pd
import torch
import torch.nn as nn

softmax = nn.Softmax(1)


def attention(q,k,v):

    dim_d = np.shape(k)[1]

 
    a = q @ np.transpose(k)
    b = np.sqrt(dim_d)
    return softmax(a/b)*v


