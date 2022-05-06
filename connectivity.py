#!/usr/bin/env python3

import mne
from mne.preprocessing import ICA
from mne.decoding import Scaler
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
import numpy as np
import mne_connectivity
from integration import get_data
from PMI import *

#	
#data, label = get_data()
#
#data = np.swapaxes(data, 1, 2)
#
#np.save('eeg_emg_data.npy', data)
#np.save('eeg_emg_label.npy', label)

def get_connectivity(data):
	connectivity = mne_connectivity.vector_auto_regression(data)
	return connectivity.get_data()

data = np.load('healthy_EEG_EMG_pull_push_data.npy')

print(data.shape)

x = SPMI_epochs(data, 5, 1)

print(x)
print(x.shape)

np.save('SPMI_healthy_data.npy', x)