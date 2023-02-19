import mne
import matplotlib.pyplot as plt

electrode_names = ['Fp1','AF3','F3','F7','FC1','FC5','C3','T7','CP1','CP5', 
                   'P3','P7','PO3','O1', 'Fz','Cz','Pz','Oz','Fp2','AF4','F4','F8',
                   'FC2','FC6','C4','T8','CP2','CP6','P4','P8','PO4','O2']

standard_montage = mne.channels.make_standard_montage('standard_1020')

poss = standard_montage.get_positions()['ch_pos']

print(poss)

#poss['F1']

print(standard_montage.ch_names)

#print({name: standard_montage.ch_names[name] for name in electrode_names})

# Get the positions of the specified electrodes from the standard montage
ch_pos = {name: poss[name] for name in electrode_names}

# Create a custom montage with the specified electrodes and their positions
custom_montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')

# Plot the custom montage
mne.viz.plot_montage(custom_montage, show_names=True)

plt.show()