#!/usr/bin/env python3

import emggao
import numpy as np

from Data_prepare import load_gdffile

def get_data():

	emgdata, emglabel = emggao.get_dataset(1)
	
	print(emgdata.shape)
	print(emglabel.shape)
	
	eegdata, eeglabel = load_gdffile.load_gdffile()
	
	eegdata = eegdata.transpose(0,2,1)
	
	print(eegdata.shape)
	print(eeglabel.shape)
	
	print(eeglabel)
	
	emg_data_class1 = []
	
	emg_data_class2 = []
	
	eeg_data_class1 = []
	
	eeg_data_class2 = []
	
	for i in range(400):
		if emglabel[i] == 0:
			emg_data_class1.append(emgdata[i])
		else:
			emg_data_class2.append(emgdata[i])
	
	for i in range(143):
		if eeglabel[i] == 7:
			eeg_data_class1.append(eegdata[i])
		else:
			eeg_data_class2.append(eegdata[i])
			
	len1 = len(eeg_data_class1)
	len2 = len(eeg_data_class2)
			
	all_data = []
	all_label = []
	
	for i in range(40):
		all_data.append(np.concatenate((eeg_data_class1[i], emg_data_class1[i]), axis=1))
		all_label.append(0)
		
	for i in range(40):
		all_data.append(np.concatenate((eeg_data_class2[i], emg_data_class2[i]), axis=1))
		all_label.append(1)
	
	all_data = np.array(all_data)
	
	all_label = np.array(all_label)
	
	print(all_data.shape)
	print(all_label.shape)
	
	return all_data, all_label


data, label = get_data()