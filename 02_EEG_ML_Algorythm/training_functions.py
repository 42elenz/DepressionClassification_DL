import os
import random
import string
import functions as f
import mne
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import LeakyReLU
import csv
import pandas as pd


def set_directories(customdir=None):
	if customdir is None:
		dataDir = os.getenv('DATA_DIR', '/platform/data')
		modelDir = os.getenv('MODEL_DIR', '/platform/data')
	else:
		dataDir=customdir
	return dataDir, modelDir

def set_peers_epochs(default_max_epochs,default_min_peers):
	max_epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
	min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))
	return max_epochs, min_peers

def generate_task_names(number_of_names=1):
	list_of_names = []
	for i in range(number_of_names):
		list_of_names.append(''.join(random.choice(string.ascii_letters) for i in range(10)))
	return list_of_names

def load_data_from_array(dataDir, group_label):
    data = []
    # iterate trough the files in the directory and load the data
    for i, filename in enumerate(os.scandir(dataDir)):
        data_array = np.load(filename)
        if group_label == []:
            group_label.extend([i] * len(data_array))
        else:
            count = np.max(group_label)
            group_label.extend([count + 1] * len(data_array))
        data.append(data_array)
    return group_label, data


def load_data(
    dataDir, group_label, sliding_window_size, sliding_window_overlap, from_array=False
):
    # load the data
    directory = dataDir
    data = []
    # iterate over files in
    # that directory
    if from_array == False:
        for i, filename in enumerate(os.scandir(directory)):
            if filename.is_file():
                with open("data_shape.txt", "a") as fil:
                    print(filename.path + " " + str(i), file=fil)
                raw_h = mne.io.read_raw_edf(filename.path, preload=True)
                raw_h.drop_channels(
                    [
                        "EEG A2-A1",
                        "EEG 23A-23R",
                        "EEG 24A-24R",
                        "EEG T3-LE",
                        "EEG T5-LE",
                        "EEG T4-LE",
                        "EEG T6-LE",
                        "EEG Cz-LE",
                        "EEG Pz-LE",
                    ],
                    on_missing="ignore",
                )
                raw_h = raw_h.rename_channels(
                    {
                        "EEG Fp1-LE": "Fp1",
                        "EEG F3-LE": "F3",
                        "EEG C3-LE": "C3",
                        "EEG P3-LE": "P3",
                        "EEG O1-LE": "O1",
                        "EEG F7-LE": "F7",
                        "EEG Fz-LE": "Fz",
                        "EEG Fp2-LE": "Fp2",
                        "EEG F4-LE": "F4",
                        "EEG C4-LE": "C4",
                        "EEG P4-LE": "P4",
                        "EEG O2-LE": "O2",
                        "EEG F8-LE": "F8",
                    }
                )
                raw_filter = f.filter_raw(raw_h, reference="average")
                raw_montage = f.make_montage(raw_filter)
                ica_filter = f.ica_filter(raw_montage, n_components=0.99)
                data_array = f.sliding_window_eeg_manually(
                    ica_filter,
                    sliding_window_size,
                    sliding_window_overlap,
                    z_normalize=True,
                    normalize=False,
                )
                if group_label == []:
                    group_label.extend([i] * len(data_array))
                else:
                    count = np.max(group_label)
                    group_label.extend([count + 1] * len(data_array))
                data.append(data_array)
    else:
        group_label, data = load_data_from_array(dataDir, group_label)
    return group_label, data

def control_for_shape(data_H, data_MDD):
	# Find the maximum shape along the first dimension
	max_shape = max(data_H[i].shape[0] for i in range(len(data_H)))
	max_shape = max(max_shape, max(data_MDD[i].shape[0] for i in range(len(data_MDD))))

	# Trim the data to have the same shape
	for i in range(len(data_H)):
		if data_H[i].shape[0] > max_shape:
			data_H[i] = data_H[i][:max_shape]
	
	for i in range(len(data_MDD)):
		if data_MDD[i].shape[0] > max_shape:
			data_MDD[i] = data_MDD[i][:max_shape]
	
	return data_H, data_MDD

def makeX_Y_data(data_H, data_MDD):
	concat_windows_H = np.concatenate(data_H, axis=0)
	concat_windows_MDD = np.concatenate(data_MDD, axis=0)
	X_data = np.concatenate((concat_windows_H, concat_windows_MDD), axis=0)

	labels_H = np.zeros(concat_windows_H.shape[0])
	labels_MDD = np.ones(concat_windows_MDD.shape[0])
	y_data = np.concatenate((labels_H,labels_MDD),axis=0)
	return(X_data, y_data)

def permutate(X_data, y_data, group_label):
	#Permutate and shuffel data
	perm = np.random.permutation(len(y_data))
	X_data = X_data[perm]
	y_data = y_data[perm]
	group_label = np.array(group_label)[perm]
	return(X_data, y_data, group_label)

def stratified_group_k_fold(X_data, y_data, group_label, test_set=True):
	sgkf = StratifiedGroupKFold(n_splits=5)
	for train_idx, test_idx in sgkf.split(X_data, y_data, group_label):
		# train and evaluate your model on this fold
		X_train_all, y_train_all = X_data[train_idx], y_data[train_idx]
		X_test, y_test = X_data[test_idx], y_data[test_idx]
		groups_train_all, groups_test = group_label[train_idx], group_label[test_idx]
	if test_set:
		sgkf = StratifiedGroupKFold(n_splits=5)
		for train_idx, val_idx in sgkf.split(X_train_all, y_train_all, groups_train_all):
		# train and evaluate your model on this fold
			X_train, y_train = X_train_all[train_idx], y_train_all[train_idx]
			X_val, y_val = X_train_all[val_idx], y_train_all[val_idx]
			groups_train, groups_val = groups_train_all[train_idx], groups_train_all[val_idx]
	else:
		X_train, y_train = X_train_all, y_train_all
		X_val, y_val = X_test, y_test
		groups_val = groups_test
		groups_train = groups_train_all
		return(groups_train, groups_val, X_train, X_val, y_train, y_val)
	return(groups_train, groups_val, groups_test, X_train, X_val, X_test, y_train, y_val, y_test)

def expand_dim(list_of_X, list_of_y):
	expanded_X=[]
	for X in list_of_X:
		X = np.expand_dims(X, axis=-1)
		expanded_X.append(X)
	expanded_Y=[]
	for y in list_of_y:
		y = tf.reshape(y,(-1,1))
		expanded_Y.append(y)
	return(expanded_X, expanded_Y)

def model_architecture(architecture_shape):
	model = keras.Sequential()
	model.add(keras.layers.Conv2D(128, kernel_size=(1,5), activation=LeakyReLU(), input_shape=architecture_shape, strides=(1,2)))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D(pool_size=(1,2), strides=(1,2)))

	model.add(keras.layers.Conv2D(64, kernel_size=(1,5), activation=LeakyReLU(), strides=(1,2)))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D(pool_size=(1,2), strides=(1,2)))

	model.add(keras.layers.Conv2D(64, kernel_size=(1,5), activation=LeakyReLU(), strides=(1,2)))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D(pool_size=(1,2), strides=(1,2)))

	model.add(keras.layers.Conv2D(32, kernel_size=(1,3), activation=LeakyReLU(), strides=(1,2)))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D(pool_size=(1,2), strides=(1,2)))

	model.add(keras.layers.Conv2D(32, kernel_size=(1,2), activation=LeakyReLU(), strides=(1,2)))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D(pool_size=(1,1), strides=(1,2)))

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(16, activation=LeakyReLU()))
	model.add(keras.layers.Dense(8, activation=LeakyReLU()))
	model.add(keras.layers.Dense(1, activation='sigmoid'))
	return(model)

def write_to_csv(data, filename):
	# Write data to csv file
	with open(filename, 'a') as f:
		for i in data:
			if i == '\n':
				f.write(str(i))
			else:
				f.write(str(i) + ',')