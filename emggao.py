#!/usr/bin/env python3

import numpy as np
import mne
import sklearn
from sklearn.model_selection import train_test_split
import lightgbm as lgb

FILTERED_DATA_ROOT = 'sEMG-dataset/filtered/csv/'

filtered_filenames = [FILTERED_DATA_ROOT + str(i) + '_filtered.csv' for i in range(1, 41)]

#data = np.genfromtxt(filtered_filenames[0], delimiter=',')
#
#print(data.shape)
#print(data)

def segmentation(data):
	fs = 2000
	signal_segment_starting=0
	signal_segment_ending=6
	
	rep_coeff = [4, 138, 272, 406, 540]
	labels_meaning = ['REST', 'EXTENSION', 'FLEXION', 'ULNAR DEVIATION', 'RADIAL DEVIATION', 'GRIP', 'ABDUCTION', 'ADDUCTION', 'SUPINATION', 'PRONATION']
	
	datas = []
	labels = []
	
	for trail in range(5):
		for gesture in range(10):
#			for channel in range(4):
			l = (signal_segment_starting+rep_coeff[trail]+(gesture*10))*fs
			r = ((rep_coeff[trail]+(gesture*10))+signal_segment_ending)*fs
			sEMG_data=data[l:r, :]
			if gesture == 0 or gesture == 1:
				datas.append(sEMG_data.flatten())
				if gesture == 1:
					labels.append(1)
				else:
					labels.append(gesture)
			
	return np.array(datas), np.array(labels)


def segmentation_downsample(data):
	fs = 2000
	signal_segment_starting=0
	signal_segment_ending=6
	
	rep_coeff = [4, 138, 272, 406, 540]
	labels_meaning = ['REST', 'EXTENSION', 'FLEXION', 'ULNAR DEVIATION', 'RADIAL DEVIATION', 'GRIP', 'ABDUCTION', 'ADDUCTION', 'SUPINATION', 'PRONATION']
	
	datas = []
	labels = []
	
	for trail in range(5):
		for gesture in range(10):
			l = (signal_segment_starting+rep_coeff[trail]+(gesture*10))*fs
			r = ((rep_coeff[trail]+(gesture*10))+signal_segment_ending)*fs
			sEMG_data=data[l:r, :]
			for i in range(8):
				datas.append(sEMG_data[i::8,:])
				labels.append(gesture)
					
	return np.array(datas), np.array(labels)



def get_dataset(id):
	data = np.genfromtxt(filtered_filenames[id-1], delimiter=',')
	temp_data, temp_label = segmentation_downsample(data)
	return temp_data, temp_label


def generate_dataset():
	data_all = []
	label_all = []
	for filename in filtered_filenames:
		data = np.genfromtxt(filename, delimiter=',')
		temp_data, temp_label = segmentation(data)
		data_all.append(temp_data)
		label_all.append(temp_label)
	data_all = np.concatenate(data_all, axis=0)
	label_all = np.concatenate(label_all, axis=0)
	return data_all, label_all

#data_all, label_all = generate_dataset()

data_id, label_id = get_dataset(1)

print(data_id.shape)

print(label_id.shape)

np.savetxt("data.csv", data_all, delimiter=",")
np.savetxt("label.csv", label_all, delimiter=",")

data_all = np.loadtxt("data.csv", delimiter=",")
label_all = np.loadtxt("label.csv", delimiter=",")

print(data_all.shape)
print(label_all.shape)

X_train, X_test, y_train, y_test = train_test_split(data_all, label_all, test_size=0.4, random_state=0)

#X_train = train_x.values
#X_test = test_x.values
#y_train = train_y.values
#y_test = test_y.values

params = {'num_leaves': 60, #结果对最终效果影响较大，越大值越好，太大会出现过拟合
	'min_data_in_leaf': 30,
	'objective': 'binary', #定义的目标函数
	'max_depth': -1,
	'learning_rate': 0.03,
	"min_sum_hessian_in_leaf": 6,
	"boosting": "gbdt",
	"feature_fraction": 0.9,  #提取的特征比率
	"bagging_freq": 1,
	"bagging_fraction": 0.8,
	"bagging_seed": 11,
	"lambda_l1": 0.1,             #l1正则
	# 'lambda_l2': 0.001,     #l2正则
	"verbosity": -1,
	"nthread": -1,                #线程数量，-1表示全部线程，线程越多，运行的速度越快
	'metric': {'binary_logloss', 'auc'},  ##评价函数选择
	"random_state": 2019, #随机数种子，可以防止每次运行的结果不一致
	# 'device': 'gpu' ##如果安装的事gpu版本的lightgbm,可以加快运算
}
print('Training...')
trn_data = lgb.Dataset(X_train, y_train)
val_data = lgb.Dataset(X_test, y_test)
clf = lgb.train(params, 
	trn_data, 
	num_boost_round = 1000,
	valid_sets = [trn_data,val_data])

print('Predicting...')
y_prob = clf.predict(X_test, num_iteration=clf.best_iteration)
y_pred = [int(x+0.5) for x in y_prob]

print(y_pred)

print(y_test)

print("AUC score: {:<8.5f}".format(sklearn.metrics.accuracy_score(y_pred, y_test)))

#LightBGM: 十分类61.5%