import torch
import torch.nn as nn
import numpy as np
from models import Encoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from textnorm import TextNorm,PositionalEncoding
from torch.nn.utils.rnn import pad_sequence
encoder  = LabelEncoder()

data = pd.read_csv('test.csv')

X = data['sentence']
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


embedding_model = nn.Embedding(len(vocab),64)


model = Encoder(64,6)
model.load_state_dict(
    torch.load(
        f='weights.pt',
        weights_only=True)
)
X_embedding = embedding_model(X_tensor)
positional_encoder = PositionalEncoding(64,5000)
X_encodings = positional_encoder(X_embedding)

model.eval()
with torch.no_grad():

    output  = model(X_encodings)
    print(output)