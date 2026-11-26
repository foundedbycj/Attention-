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


embedding_model = nn.Embedding(num_embeddings=len(vocab),embedding_dim=64)



encoder = Encoder(64,6)


criterion = nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(params=  list(encoder.parameters()) +
    list(embedding_model.parameters()),lr=0.01)
positional_encoder = PositionalEncoding(64,5000)

epochs = 10

for epoch in range (epochs):
    X_embedding = embedding_model(X_tensor)
    X_encodings = positional_encoder(X_embedding)



    encoder.train()
    embedding_model.train()
    output =encoder(X_encodings)
    y_pred = output
    y = torch.squeeze(y)
    
    
    optimiser.zero_grad()

    loss = criterion(y_pred,y)
    loss.backward()
    optimiser.step()

    print(f'loss at epoch {epoch+1} is {loss.item()}')

    torch.save(
        encoder.state_dict(),
        f='weights.pt'
    )




print (y_pred.shape , y.shape)

