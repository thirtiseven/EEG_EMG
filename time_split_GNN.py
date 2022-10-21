import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential, Flatten, Conv1d, MaxPool1d

import joblib

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import OneHotDegree
from torch_geometric.data import Data

import numpy as np
import mne
from mne.decoding import CSP

from sklearn.model_selection import train_test_split

from time_split import time_split

import random

from single_channel_process import traditional_features

from PMI import *

data_all = np.load('healthy_EEG_EMG_pull_push_data.npy')
label_all = np.load('healthy_EEG_EMG_pull_push_label.npy')

data_train, data_test, label_train, label_test = train_test_split(data_all, label_all, test_size=0.2, random_state=42)

data_train, label_train = time_split(data_train, label_train)
data_test, label_test = time_split(data_test, label_test)

#graph_data = np.load('SPMI_healthy_data.npy')
#graph_data = PMI_1epoch(data_train[0], 5, 1)

print(data_all)
print(label_all)

import random
index = [i for i in range(len(data_train))]
random.shuffle(index)
data_train = data_train[index]
label_train = label_train[index]

def get_graphs(X, y):
    graphs = []
    
    channel_tot = 40
    
    for i in range(y.shape[0]):
        if i % 100 == 0:
            print(i)
        feature = []
        cnt = 0
        for data in X[i]:
            cnt += 1
            feature.append(traditional_features(data.reshape(1,-1), channel_id=cnt))
        feature = np.array(feature)
        x = torch.tensor(feature, dtype=torch.float).reshape(40,-1)
        
        edges = []
        edge_weight = []
        
        sumee, sumem, summm, sumall = [], [], [], []
        
        graph_data = SPMI_1epoch(X[i], 5, 1)
        
        for j in range(channel_tot):
            for k in range(channel_tot):
                if graph_data[j,k] > 0:
                    sumall.append(graph_data[j,k])
                if (k < 32 and j >= 32) or (k >= 32 and j < 32):
                    if graph_data[j,k] > 0:
                        sumem.append(graph_data[j,k])
                if (k < 32 and j < 32):
                    if graph_data[j,k] > 0:
                        sumee.append(graph_data[j,k])
                if (k >= 32 and j >= 32):
                    if graph_data[j,k] > 0:
                        summm.append(graph_data[j,k])
                        
        avgee = np.percentile(sumee, 75)
        avgem = np.percentile(sumem, 75)
        avgmm = np.percentile(summm, 75)
        avgall = np.percentile(sumall, 75)
        
        label = torch.tensor([y[i]]).reshape(1,)
        for j in range(40):
            for k in range(40):
                if graph_data[j,k] > avgall:
                    edges.append([j, k])
        edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        g = Data(x=x, edge_index=edges, y=label)
    #   print(g)
        graphs.append(g)
        
    return graphs
    

test_dataset = get_graphs(data_test, label_test)
train_dataset = get_graphs(data_train, label_train)

joblib.dump(test_dataset, 'time_split_graphs_test.sav')
joblib.dump(train_dataset, 'time_split_graphs_train.sav')

test_loader = DataLoader(test_dataset, batch_size=128)
train_loader = DataLoader(train_dataset, batch_size=128)
    

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
model = Net(7, 64, 12, num_layers=2)
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

torch.save(model.state_dict(), 'time_split_model.pth')

#all = 0
#cor = 0

#cnt = 0
#cur = []

#seg_cor = 0
#seg_all = 0

@torch.no_grad()
def vali():
    all = 0
    cor = 0

    cnt = 0
    cur = []

    seg_cor = 0
    seg_all = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.max(dim=1)[1]
#   print(out)
#   print(pred)
#   print(data.y)
        for (x, y) in zip(pred, data.y):
            if (x < 6 and y < 6) or (x >= 6 and y >= 6):
                cor += 1
            all += 1
            cnt += 1
            cur.append(int(x >= 6))
            if cnt == 6:
                cnt = 0
                seg_all += 1
                seg_p = np.argmax(np.bincount(cur))
                if (y >= 6 and seg_p == 1) or (y < 6 and seg_p == 0):
                    seg_cor += 1
                cur = []
    
    print(cor/all)
    print(seg_cor/seg_all)

vali()


