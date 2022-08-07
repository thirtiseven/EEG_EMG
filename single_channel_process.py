import numpy as np
import mne

#iter_freqs = [
#	{'name': 'Delta', 'fmin': 0, 'fmax': 3.75},
#	{'name': 'Theta', 'fmin': 3.75, 'fmax': 7.5},
#	{'name': 'Alpha', 'fmin': 7.5, 'fmax': 12.5},
#	{'name': 'Beta',  'fmin': 12.5, 'fmax': 35},
#]

#iter_freqs = [
#	{'name': 'a', 'fmin': 0, 'fmax': 5},
#	{'name': 'b', 'fmin': 5, 'fmax': 10},
#	{'name': 'c', 'fmin': 10, 'fmax': 15},
#	{'name': 'd',  'fmin': 15, 'fmax': 20},
#	{'name': 'e',  'fmin': 20, 'fmax': 25},
#	{'name': 'f',  'fmin': 25, 'fmax': 35},
#	{'name': 'g',  'fmin': 30, 'fmax': 35},
#	{'name': 'h',  'fmin': 35, 'fmax': 40},
#]

iter_freqs = [
	{'name': 'a', 'fmin': 2, 'fmax': 4},
	{'name': 'b', 'fmin': 4, 'fmax': 8},
	{'name': 'c', 'fmin': 8, 'fmax': 15},
	{'name': 'd',  'fmin': 15, 'fmax': 30},
	{'name': 'e',  'fmin': 30, 'fmax': 40},
]

#iter_freqs = [
#	{'name': 'a', 'fmin': 0, 'fmax': 40},
##	{'name': 'b',  'fmin': 20, 'fmax': 40},
#]


def traditional_features(signal, channel_id=0):
	info = mne.create_info(1, 250, ch_types='eeg')
	raw = mne.io.RawArray(signal, info, verbose=False)
	psds, freqs = mne.time_frequency.psd_multitaper(raw, n_jobs=1, fmin=0, fmax=40, picks='eeg', verbose=False)
#	print(psds)
#	print(freqs)
	psds = np.squeeze(np.average(psds, axis=0))
	eventEnergy =[]
	for iter_freq in iter_freqs:
		eventEnergy.append(np.sum(psds[(iter_freq['fmin'] < freqs) & (freqs < iter_freq['fmax'])]))
#	print(eventEnergy)
	sum = np.sum(eventEnergy)
#	print(sum)
	eventEnergy = eventEnergy / sum * 100
	eventEnergy = np.append(eventEnergy, sum/len(iter_freq)*1e7)
	eventEnergy = np.append(eventEnergy, channel_id)
#	print(eventEnergy)
	return eventEnergy

#data_all = np.load('healthy_EEG_EMG_pull_push_data.npy')
#
#print(data_all.shape)
#
#for i in range(40):
#	traditional_features(data_all[0][i].reshape(1,1501))