low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 32.  # high cut frequency for filtering
t_max = 6.995   #  eopch time end
t_min = 1.0  #  eopch time begin

Competition4_filename = 'A01T'  #gdffile
Competition4_cueLeft = 7   #left mark
Competition4_cueRight = 8   #right mark
Competition4_filepath = '/Users/lihaoyang/Code/BCI_Framework/Data/BCICIV_2a_gdf/'
Competition4_eventDescription = {'276': "eyesOpen", '277': "eyesClosed", '768': "startTrail", '769': "cueLeft",
                        '770': "cueRight", '771': "cueFoot", '772': "cueTongue", '783': "cueUnknown",
                        '1023': "rejected", '1072': 'eye movements', '32766': "startRun"}
Competition4_eog = ['EOG-left', 'EOG-central', 'EOG-right']
Competition4_channel_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
Competition4_channel_names = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3',
                'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-C11',
                'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16']
Competition4_epoch_choose = ['cueLeft', 'cueRight']


CNN_sample = 100
CNN_feature = 876
CNN_batch = 10
CNN_test = 44
CNN_channel_count = 22
CNN_label_reform = 7
