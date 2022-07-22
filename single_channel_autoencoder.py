import torch
from torch import nn

import copy

import numpy as np
import mne

import random

from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_all = np.load('healthy_EEG_EMG_pull_push_data.npy')

data_all = data_all * 10000

print(data_all.shape)

seq_len = 1501
n_features = 1

data = data_all.reshape(20640,-1)[:10,0:seq_len]
	
print(data.shape)

class Encoder(nn.Module):
	
	def __init__(self, seq_len, n_features, embedding_dim=64):
		super(Encoder, self).__init__()

		self.seq_len, self.n_features = seq_len, n_features
		self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

		self.rnn1 = nn.LSTM(
			input_size=n_features,
			hidden_size=self.hidden_dim,
			num_layers=1,
			batch_first=True
		)

		self.rnn2 = nn.LSTM(
			input_size=self.hidden_dim,
			hidden_size=embedding_dim,
			num_layers=1,
			batch_first=True
		)

	def forward(self, x):
		x = x.reshape((1, self.seq_len, self.n_features))
		
		x, (_, _) = self.rnn1(x)
		x, (hidden_n, _) = self.rnn2(x)
		
		return hidden_n.reshape((self.n_features, self.embedding_dim))
	
	
class Decoder(nn.Module):

	def __init__(self, seq_len, input_dim=64, n_features=1):
		super(Decoder, self).__init__()
		self.seq_len, self.input_dim = seq_len, input_dim
		self.hidden_dim, self.n_features = 2 * input_dim, n_features

		self.rnn1 = nn.LSTM(
			input_size=input_dim,
			hidden_size=input_dim,
			num_layers=1,
			batch_first=True
		)

		self.rnn2 = nn.LSTM(
			input_size=input_dim,
			hidden_size=self.hidden_dim,
			num_layers=1,
			batch_first=True
		)

		self.output_layer = nn.Linear(self.hidden_dim, n_features)
		
	def forward(self, x):
		x = x.repeat(self.seq_len, self.n_features)
		x = x.reshape((self.n_features, self.seq_len, self.input_dim))
		
		x, (hidden_n, cell_n) = self.rnn1(x)
		x, (hidden_n, cell_n) = self.rnn2(x)
		x = x.reshape((self.seq_len, self.hidden_dim))
		
		return self.output_layer(x)
	
class RecurrentAutoencoder(nn.Module):
	def __init__(self, seq_len, n_features, embedding_dim=64):
		super(RecurrentAutoencoder, self).__init__()

		self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
		self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x

model = RecurrentAutoencoder(seq_len, n_features, 32)
model = model.to(device)

def train_model(model, train_dataset, val_dataset, n_epochs):
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#	criterion = nn.L1Loss(reduction='sum').to(device)
	criterion = nn.MSELoss().to(device)
	history = dict(train=[], val=[])
	best_model_wts = copy.deepcopy(model.state_dict())
	best_loss = 10000.0

	for epoch in range(1, n_epochs + 1):
		print("now epoch", epoch)
		model = model.train()

		train_losses = []
		for seq_true in train_dataset:
			optimizer.zero_grad()
			seq_true = seq_true.to(device)
			seq_pred = model(seq_true)
			loss = criterion(seq_pred, seq_true)
			loss.backward()
			optimizer.step()
	
			train_losses.append(loss.item())
	
		val_losses = []
		model = model.eval()
		with torch.no_grad():
			for seq_true in val_dataset:
	
				seq_true = seq_true.to(device)
				seq_pred = model(seq_true)
	
				loss = criterion(seq_pred, seq_true)
				val_losses.append(loss.item())
	
		train_loss = np.mean(train_losses)
		val_loss = np.mean(val_losses)

		history['train'].append(train_loss)
		history['val'].append(val_loss)

		if val_loss < best_loss:
			best_loss = val_loss
			best_model_wts = copy.deepcopy(model.state_dict())
			
			if epoch % 10 == 0:
				torch.save(model.state_dict(), 'model' + str(epoch))

		print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

	model.load_state_dict(best_model_wts)
	return model.eval(), history

print("split started.")
train_data, val_data = train_test_split(
	data,
	test_size=0.1,
	random_state=42
)

train_dataset = [torch.tensor(s).unsqueeze(1).float() for s in train_data]

val_dataset = [torch.tensor(s).unsqueeze(1).float() for s in val_data]

print("train started.")

model, history = train_model(
	model,
	train_dataset,
	val_dataset,
	n_epochs=100
)

torch.save(model.state_dict(), 'model')

model2 = RecurrentAutoencoder(seq_len, n_features, 64)
model2.load_state_dict(torch.load('model'))

#with torch.no_grad():
#	for seq_true in val_dataset:
#		
#		seq_true = seq_true.to(device)
#		
#		feature = model2.encoder(seq_true)
#		
#		print(feature)
#		
#		seq_pred = model2(seq_true)
#		
#		print(seq_true)
#		print(seq_pred)
		
all_dataset = [torch.tensor(s).unsqueeze(1).float() for s in data]

all_feature = []

with torch.no_grad():
	for seq_true in all_dataset:
		seq_true = seq_true.to(device)
		feature = model2.encoder(seq_true)
		all_feature.append(feature.detach().numpy())
		
all_feature_np = np.array(all_feature)

np.save('all_feature_healthy_EEG_EMG.npy', all_feature)

#x = np.load('all_feature_healthy_EEG_EMG.npy')
#
##print(x)
