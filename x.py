import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from textnorm import TextNorm , PositionalEncoding
from torch.nn.utils.rnn import pad_sequence
from models import Self_Attention , Encoder

encoder = LabelEncoder()

data = pd.read_csv('data.csv')

X = data['text']
y = data['sentiment']
y = torch.tensor(encoder.fit_transform(y=y))
y = torch.unsqueeze(y,dim=-1)

text_normaliser = TextNorm(X)

tokenised_text = text_normaliser.transform_text(X)


vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for sentence in tokenised_text :
    for word in sentence :
        if word not in vocab:
            vocab[word] = len(vocab)