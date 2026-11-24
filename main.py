import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from textnorm import TextNorm , PositionalEncoding
from torch.nn.utils.rnn import pad_sequence
from models import Self_Attention

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


embedding_model = nn.Embedding(76,64)
X_embedding = embedding_model(X_tensor)
positional_encoder = PositionalEncoding(64,5000)
X_encodings = positional_encoder(X_embedding)





self_attention = Self_Attention(64)

attention_output = self_attention(X_encodings)

print(attention_output)

norm = nn.LayerNorm(attention_output.shape)


normalised = norm(attention_output+X_encodings)

print(normalised.shape)

ffn = nn.Sequential(
    nn.Linear(64,128),
    nn.ReLU(),
    nn.Linear(128,64)
)

output = ffn(normalised)

print(output)