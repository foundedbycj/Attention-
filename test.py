import torch
import torch.nn as nn
import numpy as np
from models import Encoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from textnorm import TextNorm,PositionalEncoding
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
encoder  = LabelEncoder()
data = pd.read_csv('data.csv')


X = data['text']

y = torch.tensor(encoder.fit_transform(data['sentiment']))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)

text_normaliser = TextNorm(X)

tokenised_train = text_normaliser.transform_text(X_train)
tokenised_test  = text_normaliser.transform_text(X_test)

vocab = torch.load('vocab.pt')

def encode_text(tokenised_data, vocab):
    encoded = []

    for sentence in tokenised_data:
        ids = [
            vocab.get(word, vocab["<UNK>"])
            for word in sentence
        ]
        encoded.append(torch.tensor(ids, dtype=torch.long))

    return encoded

X_train_encoded = encode_text(tokenised_train, vocab)
X_test_encoded = encode_text(tokenised_test, vocab)


X_train_tensor = pad_sequence(
    sequences=X_train_encoded,
    batch_first=True,
    padding_value=vocab['<PAD>']
)

X_test_tensor = pad_sequence(
    sequences=X_test_encoded,
    batch_first=True,
    padding_value=vocab['<PAD>']
)
embedding_model = nn.Embedding(num_embeddings=len(vocab),embedding_dim=64)


#checkpoint

checkpoint = torch.load(
    f=  'weights.pt',
    weights_only=True
)

embedding_model = nn.Embedding(len(vocab),64)
embedding_model.load_state_dict(checkpoint['embedding']
)

model = Encoder(64,6)
model.load_state_dict(checkpoint['encoder']
)


mask = (X_test_tensor != vocab['<PAD>'])
X_embedding = embedding_model(X_test_tensor)
positional_encoder = PositionalEncoding(64,5000)
X_encodings = positional_encoder(X_embedding)

embedding_model.eval()
model.eval()
with torch.no_grad():

    softmax = nn.Softmax(dim=1)
    output  = model(X_encodings,mask)
    predictions = torch.softmax(output, dim=1)
    print(predictions.round(decimals=2),y_test)