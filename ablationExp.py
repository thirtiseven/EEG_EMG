import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential, Flatten, Conv1d, MaxPool1d

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import OneHotDegree
from torch_geometric.data import Data

import numpy as np
import mne
from mne.decoding import CSP

import random

from single_channel_process import traditional_features

data_all = np.load('healthy_EEG_EMG_pull_push_data.npy')
label_all = np.load('healthy_EEG_EMG_pull_push_label.npy')
graph_data = np.load('SPMI_healthy_data.npy')

# Load and preprocess data
# Here we randomly generate several graphs for simplicity as an example
graphs = []
for i in range(label_all.shape[0]):
    feature = []
    cnt = 0
    for data in data_all[i]:
        cnt += 1
        feature.append(traditional_features(data.reshape(1,-1), channel_id=cnt))
    x = torch.tensor(feature, dtype=torch.float).reshape(40,-1)
    
#   x = torch.tensor(features[i], dtype=torch.float).reshape(40,-1)
    edges = []
    edge_weight = []
    label = torch.tensor([label_all[i]]).reshape(1,)
    for j in range(40):
        for k in range(40):
            if graph_data[i,j,k] < 0.5:
                edges.append([j, k])
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
    g = Data(x=x, edge_index=edges, y=label)
#   print(g)
    graphs.append(g)

random.shuffle(graphs)

test_dataset = graphs[:len(graphs) // 5]
train_dataset = graphs[len(graphs) // 5:]
test_loader = DataLoader(test_dataset, batch_size=128)
train_loader = DataLoader(train_dataset, batch_size=128)

class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)
    

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for i in range(num_layers):
            mlp = Sequential(
                Linear(in_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
#           mlp = Sequential(
#               Conv1d(1, 1, kernel_size=64, stride=2, padding=1),
#               MaxPool1d(2),
#               Linear(360, hidden_channels)
#           )
            conv = GINConv(mlp, train_eps=True).jittable()
            
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))
            
            in_channels = hidden_channels
            
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
        x = global_add_pool(x, batch)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(10, 64, 2, num_layers=2)
model = model.to(device)
model = torch.jit.script(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    
    total_loss = 0.
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
#       print(data.x.shape)
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
#       print(data.y)
#       print(out)
#       break
#       loss = torch.nn.CrossEntropyLoss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    
    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.max(dim=1)[1]
        total_correct += pred.eq(data.y).sum().item()
    return total_correct / len(loader.dataset)


for epoch in range(1, 301):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')

model.eval()

for data in test_loader:
    data = data.to(device)
    out = model(data.x, data.edge_index, data.batch)
    pred = out.max(dim=1)[1]
#   print(out)
    print(pred)
    print(data.y)