import torch
from torch import nn

import copy

import numpy as np
import mne

import random

from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_all = np.load('healthy_EEG_EMG_pull_push_data.npy')

print(data_all.shape)

seq_len = 1501
n_features = 1

data = data_all.reshape(20640,-1)[:,0:seq_len]

print("split started.")
train_data, val_data = train_test_split(
    data,
    test_size=0.15,
    random_state=42
)

train_dataset = [torch.tensor(s).unsqueeze(1).float() for s in train_data]

val_dataset = [torch.tensor(s).unsqueeze(1).float() for s in val_data]

print("train started.")

model = AutoEncForecast(config, input_size=nb_features).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])