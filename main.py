import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from textnorm import TextNorm

encoder = LabelEncoder()

data = pd.read_csv('data.csv')

X = data['text']
y = data['sentiment']
y = encoder.fit_transform(y=y)


text_normaliser = TextNorm(X)

X = text_normaliser.transform_text(X)


vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for sentence in X :
    for word in sentence :
        if word not in vocab:
            vocab[word] = len(vocab)

print(vocab)