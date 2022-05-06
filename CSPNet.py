#!/usr/bin/env python3

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
#from read_data import read_the_fuck_data

#data_train, labels_train, data_test, labels_test = read_the_fuck_data(40)

data_all = np.load('healthy_EEG_EMG_pull_push_data.npy')
label_all = np.load('healthy_EEG_EMG_pull_push_label.npy')
#data_all = np.load('all_EEG_EMG_pull_push_data.npy')
#label_all = np.load('all_EEG_EMG_pull_push_label.npy')
#data_all = np.load('14EEG_EMG_pull_push_data.npy')
#label_all = np.load('14EEG_EMG_pull_push_label.npy')

data_train, data_test, labels_train, labels_test = train_test_split(data_all, label_all, test_size=0.2, random_state=0)
#from emggao import get_dataset
#
#data, labels = get_dataset(2)
#
#print(labels)
#data = np.swapaxes(data, 1, 2)

print(data_train.shape)
print(labels_train.shape)
print(data_test.shape)
print(labels_test.shape)

CSP_ncomponents = 24

sample_size = data_train.shape[2]

#feature = CSP(n_components=CSP_ncomponents, reg=None, norm_trace=True, transform_into='average_power')
feature = CSP(n_components=CSP_ncomponents, reg=None, norm_trace=True)

#scp_model = feature.get_params()
#np.save('scp_model.npy', scp_model)

print("feature")

feature_fited = feature.fit(data_train, labels_train)
features = feature.transform(data_train)

print(features.shape)

num_inputs = CSP_ncomponents
num_actions = 2
num_hidden = num_inputs*4

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
common = layers.Dropout(0.3)(common)
common = layers.Dense(num_hidden, activation="relu")(common)
common = layers.Dropout(0.3)(common)
action = layers.Dense(num_actions, activation="softmax")(common)
#critic = layers.Dense(1)(common)
#model = keras.Model(inputs=inputs, outputs=[action, critic])
model = keras.Model(inputs=inputs, outputs=action)

#model.layers[-1].trainable = False

#model.compile(loss = ['categorical_crossentropy', 'categorical_crossentropy'], optimizer = 'adam', metrics=['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

callback = keras.callbacks.EarlyStopping(monitor='val_loss',
					min_delta=0,
					patience=10,
					verbose=0, mode='auto')
					
critic_label = np.random.rand(labels_train.shape[0])
critic_label_test = np.random.rand(labels_test.shape[0])

#fittedModel = model.fit(x = features, y = [to_categorical(labels_train, 2), critic_label], epochs = 500)
fittedModel = model.fit(x = features, y = to_categorical(labels_train, 2), epochs = 500, shuffle = True)

test_features = feature.transform(data_test)

predicted = model.predict(test_features)

test_acc = model.evaluate(test_features, to_categorical(labels_test, 2))
#test_acc = model.evaluate(test_features, [to_categorical(labels_test, 2), critic_label_test])

print('test_acc: ',test_acc)

#model.layers[-1].trainable = True

#model.save("cspnet.h5")
