import numpy as np

def time_split(X, y):
	X = X[:,:,:1500]
	
	print(X.shape)
	print(y.shape)
	
	data_time_split = []
	label_time_split = []
	
	for (data, label) in zip(X, y):
		for i in range(6):
			data_time_split.append(data[:,250*i:250*(i+1)])
			label_time_split.append(label*6+i)
			
	data_time_split = np.array(data_time_split)
	label_time_split = np.array(label_time_split)
	
	print(data_time_split.shape)
	print(label_time_split.shape)
	
	return data_time_split, label_time_split

data_all = np.load('healthy_EEG_EMG_pull_push_data.npy')
label_all = np.load('healthy_EEG_EMG_pull_push_label.npy')
#

data_all = data_all[:,:,:1500]

print(data_all.shape)
print(label_all.shape)

data_time_split = []
label_time_split = []

for (data, label) in zip(data_all, label_all):
	for i in range(6):
		data_time_split.append(data[:,250*i:250*(i+1)])
		label_time_split.append(label*6+i)
		
data_time_split = np.array(data_time_split)
label_time_split = np.array(label_time_split)

print(data_time_split.shape)
print(label_time_split.shape)

np.save('healthy_EEG_EMG_pull_push_data_time_split.npy', data_time_split)
np.save('healthy_EEG_EMG_pull_push_label_time_split.npy', label_time_split)