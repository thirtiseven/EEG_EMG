import numpy as np

#print(data.shape)
#print(labels.shape)

from cogdl import experiment
from cogdl.data import Graph
from cogdl.datasets import GraphDataset
import torch

class MyGraphDataset(GraphDataset):
	def __init__(self, path="data.pt"):
		self.path = path
		super(MyGraphDataset, self).__init__(path, metric="accuracy")
		
	def process(self):
		data = np.load('vector_auto_regression_data.npy')
		labels = np.load('eeg_emg_label.npy')
		# Load and preprocess data
		# Here we randomly generate several graphs for simplicity as an example
		graphs = []
		for i in range(80):
			edges = []
			edge_weight = []
			label = torch.tensor([labels[i]]).reshape(1,)
			for j in range(26):
				for k in range(26):
					if data[i,j,k] > 0:
						edges.append([i, j])
						edge_weight.append(data[i,j,k])
			edges = torch.tensor(edges).t()
			edge_weight = torch.tensor(edge_weight)
#			print(edges)
#			print(edges.shape)
#			print(label.shape)
			g = Graph(edge_index=edges, y=label)
			g.edge_weight = edge_weight
			graphs.append(g)
			
		return graphs
	
if __name__ == "__main__":
	dataset = MyGraphDataset()
	#dataset.process()
	experiment(model="gin", dataset=dataset, cpu=True, epochs=100)