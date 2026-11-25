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


class EncoderUnit(nn.Module):

    def __init__(self,dim_model):
        super().__init__()

        self.attention = Self_Attention(dim_model)

        self.layernorm1 = nn.LayerNorm(dim_model)
        self.layernorm2 = nn.LayerNorm(dim_model)

        self.fc1        = nn.Sequential(
            nn.Linear(dim_model,128),
            nn.ReLU(),
            nn.Linear(128,dim_model)
        )

    def forward(self,x):

        attn_output = self.attention(x)
        norm1  = self.layernorm1(x+attn_output)
        FC   = self.fc1(norm1)
        norm2 = self.layernorm2(norm1+FC)

        return norm2

class Encoder(nn.Module):

    def __init__(self,dim_model,num_layer):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderUnit(dim_model)
            for _ in range(num_layer)]
        )
        self.classifier = nn.Linear(64,1)


    def forward(self,x):

        for layer in self.layers:
            x = layer(x)
            encoded = x

            prediction = self.classifier(encoded[:, 0, :])


        return(prediction)



