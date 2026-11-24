import torch.nn as nn
import torch
import numpy as np
class Self_Attention(nn.Module):
    def __init__ (self,dim_model):
        super().__init__()
        self.wQ = nn.Linear(dim_model,dim_model)
        self.wK = nn.Linear(dim_model,dim_model)
        self.wV = nn.Linear(dim_model,dim_model)

    def forward(self,X):
        Q = self.wQ(X)
        K = self.wK(X)
        V = self.wV(X)

        softmax = nn.Softmax(-1)

        scores = Q@torch.transpose(input=K,dim0=1,dim1=2)

        scores = scores/torch.sqrt(torch.tensor(K.shape[-1]))
        

        output = softmax(scores) @ V

        return output




