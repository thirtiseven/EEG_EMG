#!/usr/bin/env python3

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
#from read_data import read_the_fuck_data
from cogdl import experiment
from cogdl.data import Graph
from cogdl.datasets import GraphDataset
import cogdl.experiments as exp
import torch


class MyGraphDataset(GraphDataset):
	def __init__(self, path="data.pt"):
		self.path = path
		super(MyGraphDataset, self).__init__(path, metric="accuracy")
		
	def process(self):
		data_all = np.load('healthy_EEG_EMG_pull_push_data.npy')
		label_all = np.load('healthy_EEG_EMG_pull_push_label.npy')
		graph_data = np.load('SPMI_healthy_data.npy')
		
		CSP_ncomponents = 40
		
		sample_size = data_all.shape[2]
		
		#feature = CSP(n_components=CSP_ncomponents, reg=None, norm_trace=True, transform_into='average_power')
		feature = CSP(n_components=CSP_ncomponents, reg=None, norm_trace=True)
		
		#scp_model = feature.get_params()
		#np.save('scp_model.npy', scp_model)
		
		print("feature")
		
		feature_fited = feature.fit(data_all, label_all)
		features = feature.transform(data_all)
		
		print(features.shape)
		
		# Load and preprocess data
		# Here we randomly generate several graphs for simplicity as an example
		graphs = []
		for i in range(label_all.shape[0]):
			x = torch.tensor(features[i]).reshape(40,1)
			edges = []
			edge_weight = []
			label = torch.tensor([label_all[i]]).reshape(1,)
			for j in range(40):
				for k in range(40):
					if graph_data[0,j,k] > 0.5:
						edges.append([j, k])
			edges = torch.tensor(edges).t()
#			print(edges)
#			print(edges.shape)
#			print(label.shape)
#			print(x.shape)
			g = Graph(x=x, edge_index=edges, y=label)
			print(g)
			graphs.append(g)
#			break
			
		return graphs
	
def gao(**kwargs):
	args = kwargs
	for key, value in kwargs.items():
		args.__setattr__(key, value)
	return exp.train(args)
	
	
if __name__ == "__main__":
	dataset = MyGraphDataset()
#	dataset.process()
#	exp.experiment(model="gin", dataset=dataset, cpu=True, epochs=15, train_ratio=0.8, test_ratio=0.2)
	res = gao(model="gin", dataset=dataset, cpu=True, epochs=15, train_ratio=0.8, test_ratio=0.2)
	print(res)