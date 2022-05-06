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
		data = np.load('PMI_data.npy')
		labels = np.load('14EEG_EMG_pull_push_label.npy')
		# Load and preprocess data
		# Here we randomly generate several graphs for simplicity as an example
		graphs = []
		for i in range(labels.shape[0]):
			edges = []
			edge_weight = []
			label = torch.tensor([labels[i]]).reshape(1,)
			for j in range(40):
				for k in range(40):
					if data[i,j,k] > 0.5:
						edges.append([j, k])
						edge_weight.append(data[i,j,k])
			edges = torch.tensor(edges).t()
			edge_weight = torch.tensor(edge_weight)
			print(edges)
			print(edges.shape)
			print(label.shape)
			g = Graph(edge_index=edges, y=label)
			g.edge_weight = edge_weight
			graphs.append(g)
#			break
			
		return graphs
	
if __name__ == "__main__":
	dataset = MyGraphDataset()
	#dataset.process()
	experiment(model="gin", dataset=dataset, cpu=True, epochs=100, train_ratio=0.8, test_ratio=0.2)