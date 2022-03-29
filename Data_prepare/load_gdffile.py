import matplotlib.pyplot as plt
import numpy as np
import mne
import Data_prepare.dp_config as config
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader


def load_gdffile():
    path = config.Competition4_filepath
    filename = config.Competition4_filename

    eventDescription = config.Competition4_eventDescription

    rawDatafile = mne.io.read_raw_gdf(path + filename + ".gdf", preload=True,
                                      eog=config.Competition4_eog)
#    print(rawDatafile.info)

    ch_types = config.Competition4_channel_types
    ch_names = config.Competition4_channel_names

    # info = mne.create_info(ch_names=ch_names, sfreq=rawDatafile.info['sfreq'], ch_types=ch_types)
    #
    # data = np.squeeze(np.array([rawDatafile['EEG-Fz'][0], rawDatafile['EEG-0'][0], rawDatafile['EEG-1'][0],
    #                             rawDatafile['EEG-2'][0], rawDatafile['EEG-3'][0], rawDatafile['EEG-4'][0],
    #                             rawDatafile['EEG-5'][0], rawDatafile['EEG-C3'][0], rawDatafile['EEG-6'][0],
    #                             rawDatafile['EEG-Cz'][0], rawDatafile['EEG-7'][0], rawDatafile['EEG-C4'][0],
    #                             rawDatafile['EEG-8'][0], rawDatafile['EEG-9'][0], rawDatafile['EEG-10'][0],
    #                             rawDatafile['EEG-11'][0], rawDatafile['EEG-12'][0], rawDatafile['EEG-13'][0],
    #                             rawDatafile['EEG-14'][0], rawDatafile['EEG-Pz'][0], rawDatafile['EEG-15'][0],
    #                             rawDatafile['EEG-16'][0]]))


    picks = mne.pick_types(rawDatafile.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
    # rawData = mne.io.RawArray(data, info)

    # 未设置  montage
    # rawData.set_montage(mne.channels.make_standard_montage('biosemi64'))

    event, _ = mne.events_from_annotations(rawDatafile)
    event_id = {}
    for i in _:
        event_id[eventDescription[i]] = _[i]
#    print(event_id)

    epochs = mne.Epochs(rawDatafile, event, event_id, tmax=config.t_max, tmin=config.t_min, event_repeated='merge',
                        baseline=None, preload=True, proj=True, picks=picks)
    epochs_train = epochs[config.Competition4_epoch_choose].copy().filter(config.low_cut_hz,
                                                               config.high_cut_hz, fir_design='firwin',
                                                               skip_by_annotation='edge')

    labels = epochs_train.events[:, -1]
    epochs_data_train = epochs_train.get_data()
#    print(epochs_data_train.shape)
    # epochs_train.plot()
    # plt.show()

    # 未设置  montage
    # rawDatafile.plot_sensors()

    # return epochs_data_train, labels, rawDatafile.info
    return epochs_data_train, labels


def load_gdffile_cnn():
    sample_count = config.CNN_sample
    feature_count = config.CNN_feature
    batch_size = config.CNN_batch
    test_sample_count = config.CNN_test
    channel_count = config.CNN_channel_count
    label_reform = config.CNN_label_reform

    # import train data
    epochs_data_train, labels = load_gdffile()
    X = np.zeros((sample_count, channel_count, feature_count))

    # reform label 7/8 to label 0/1
    Y = labels[0:sample_count] - label_reform
    # print(Y)
    X = epochs_data_train[0:sample_count, :, :]

    # print(X, Y)
    train_dataset = Data.TensorDataset(torch.tensor(X), torch.tensor(Y))
    print(torch.tensor(X).shape)

    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    # print (train_dataset,train_loader)

    # import test data
    X = np.zeros((test_sample_count, channel_count, feature_count,))

    # reform label 7/8 to label 0/1
    Y = labels[sample_count:] - label_reform
    X = epochs_data_train[sample_count:, :, :]


    test_dataset = Data.TensorDataset(torch.tensor(X), torch.tensor(Y))
    print(torch.tensor(X).shape)

    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, len(train_dataset), test_loader, len(test_dataset)


if __name__ == '__main__':
    #load_gdffile()
    load_gdffile_cnn()