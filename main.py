import numpy as np
import pandas as pd
def softmax(x, axis=-1):
    """
    Compute the softmax of vector/matrix x in a numerically stable way.
    
    Parameters:
    x (array-like): Input values (logits)
    axis (int): The axis along which the softmax will be computed
    """
    # 1. Convert input to a numpy array
    x = np.asarray(x)
    
    # 2. Subtract the maximum value along the axis to prevent overflow (e^x getting too large)
    # keepdims=True ensures the dimensions match for proper array broadcasting
    max_x = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)
    
    # 3. Divide by the sum along the axis to get values between 0 and 1 that total 1
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention(q,k,v):

    dim_d = np.shape(k)[1]

 
    a = q @ np.transpose(k)
    b = np.sqrt(dim_d)

