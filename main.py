import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

data = pd.read_csv('data.csv')
X = data['text']
y = data['sentiment']

y = encoder.fit_transform(y=y)

print(y)