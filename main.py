import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from textnorm import TextNorm
from torch.nn.utils.rnn import pad_sequence

encoder = LabelEncoder()

data = pd.read_csv('data.csv')

X = data['text']
y = data['sentiment']
y = encoder.fit_transform(y=y)


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

encoded_text = []

for sentence in tokenised_text:
    encoded_sentence = []
    for word in sentence:
        encoded_sentence.append(
            vocab.get(word,vocab["<UNK>"])
        )
    encoded_text.append(encoded_sentence)

sequences = []
for sentence in encoded_text:
    tensors = torch.tensor(sentence,dtype=torch.long)
    sequences.append(tensors)

X_tensor = pad_sequence(
    sequences=sequences,
    batch_first=True,
    padding_value=vocab['<PAD>']
)


embedding_model = nn.Embedding(75,2)
X_embedding = embedding_model(X_tensor)

print(X_embedding)