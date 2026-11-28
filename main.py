import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from textnorm import TextNorm , PositionalEncoding
from torch.nn.utils.rnn import pad_sequence
from models import Self_Attention , Encoder
from sklearn.model_selection import train_test_split
encoder = LabelEncoder()

data = pd.read_csv('data.csv')


X = data['text']

y = torch.tensor(encoder.fit_transform(data['sentiment']))
y = torch.unsqueeze(y,-1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)

text_normaliser = TextNorm(X)

tokenised_train = text_normaliser.transform_text(X_train)
tokenised_test  = text_normaliser.transform_text(X_test)

vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for sentence in tokenised_train :
    for word in sentence :
        if word not in vocab:
            vocab[word] = len(vocab)

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



encoder = Encoder(64,6)


criterion = nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(params=  list(encoder.parameters()) +
    list(embedding_model.parameters()),lr=0.0008)
positional_encoder = PositionalEncoding(64,5000)

epochs = 100

for epoch in range (epochs):

    valid_mask =( X_train_tensor != vocab['<PAD>'])
    X_embedding = embedding_model(X_train_tensor)
    X_encodings = positional_encoder(X_embedding)


    encoder.train()
    embedding_model.train()
    output =encoder(X_encodings,valid_mask)
    y_pred = output
    y_train = torch.squeeze(y_train)
    optimiser.zero_grad()

    loss = criterion(y_pred,y_train)
    loss.backward()
    optimiser.step()

    print(f'loss at epoch {epoch+1} is {loss.item()}')

    torch.save(
        {'encoder':encoder.state_dict(),
        'embedding': embedding_model.state_dict() },
        f='weights.pt'
    )





torch.save(
    vocab,
    'vocab.pt'
)