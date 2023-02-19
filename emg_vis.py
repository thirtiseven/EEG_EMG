import numpy as np
import mne
import matplotlib.pyplot as plt

# load numpy array from file
data = np.load('healthy_EMG_train_data.npy')

# select first 8x1500 part of the data
emg_data = data[63,:,:]

# create mne raw object
info = mne.create_info(ch_names=['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8'], sfreq=500)
raw = mne.io.RawArray(emg_data, info)

# plot the data
raw.plot(n_channels=8, title='EMG Data')

plt.show()