import functions as f
import mne
import os
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import LeakyReLU
import csv
import pandas as pd
from numpy.random import seed
seed(4)
from tensorflow import random
random.set_seed(2)

def control_for_shape(data_H, data_MDD):
	# Find the maximum shape along the first dimension
	min_shape = min(data_H[i].shape[0] for i in range(len(data_H)))
	min_shape = min(min_shape, min(data_MDD[i].shape[0] for i in range(len(data_MDD))))

	# Trim the data to have the same shape, trim the group label list accordingly
	for i in range(len(data_H)):
		if data_H[i].shape[0] > min_shape:
			data_H[i] = data_H[i][:min_shape]
	
	for i in range(len(data_MDD)):
		if data_MDD[i].shape[0] > min_shape:
			data_MDD[i] = data_MDD[i][:min_shape]
	
	return data_H, data_MDD


def print_data_tofile(data):
	#print shape of data which is a list to a file
	with open('data_shape.txt', 'a') as f:
		print('######################################', file=f)
		for i in range(len(data)):
			print(data[i].shape, file=f)


def set_counter(group_label):
	#check if list is empty
	if group_label == []:
		counter = 0
	else:
		counter = max(group_label) + 1
	return counter

def load_data(dataDir,group_label, sliding_window_size, sliding_window_overlap):
	#load the data
	directory = dataDir
	data = []
	counter = set_counter(group_label)
	error_files = []
	# iterate over files in
	# that directory

	for i, filename in enumerate(os.scandir(directory)):
		if filename.is_file():
			try:
				with open('data_shape.txt', 'a') as fil:
					print(filename.path + ' ' + str(i), file=fil)
				raw_h = mne.io.read_raw_edf(filename.path, preload=True)
				raw_h.drop_channels(['EEG A2-A1', 'EEG 23A-23R', 'EEG 24A-24R', 'EEG T3-LE', 'EEG T5-LE', 'EEG T4-LE', 'EEG T6-LE', 'EEG Cz-LE', 'EEG Pz-LE'], on_missing='ignore')
				raw_h = raw_h.rename_channels({'EEG Fp1-LE': 'Fp1', 'EEG F3-LE': 'F3', 'EEG C3-LE': 'C3', 'EEG P3-LE': 'P3', 'EEG O1-LE': 'O1', 'EEG F7-LE': 'F7', 'EEG Fz-LE': 'Fz', 'EEG Fp2-LE': 'Fp2', 'EEG F4-LE': 'F4', 'EEG C4-LE': 'C4', 'EEG P4-LE': 'P4', 'EEG O2-LE': 'O2', 'EEG F8-LE': 'F8'})
				raw_filter = f.filter_raw(raw_h, reference='average')
				raw_montage = f.make_montage(raw_filter)
				ica_filter = f.ica_filter(raw_montage, n_components=0.99)
				data_array = f.sliding_window_eeg_manually(ica_filter, sliding_window_size, sliding_window_overlap, z_normalize=True, normalize=False)
				group_label.extend([counter + i] * len(data_array))
				data.append(data_array)
			except Exception as e:
				print(f"Error occurred with file: {filename.path}")
				print(e)
				error_files.append(filename.path)

	# Now error_files will contain the list of filenames that caused errors
	#print("ERROR FILE", error_files)
	return group_label, data

def makeX_Y_data(data_H, data_MDD):
	concat_windows_H = np.concatenate(data_H, axis=0)
	concat_windows_MDD = np.concatenate(data_MDD, axis=0)
	X_data = np.concatenate((concat_windows_H, concat_windows_MDD), axis=0)

	labels_H = np.zeros(concat_windows_H.shape[0])
	labels_MDD = np.ones(concat_windows_MDD.shape[0])
	y_data = np.concatenate((labels_H,labels_MDD),axis=0)
	return(X_data, y_data)

def permutate(X_data, y_data, group_label,random_state):
	#Permutate and shuffel data
	seed(random_state)
	random.set_seed(random_state)
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
		return(groups_train, groups_test, X_train, X_val, y_train, y_val)
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

def add_metrics_to_csv(window_size, window_overlap, loss, accuracy, precision, recall, auc, groups_train, groups_val, groups_test, file_path='metrics.csv'):
	data = {
		'Window Size': [window_size],
		'Window Overlap': [window_overlap],
		'Loss': [loss],
		'Accuracy': [accuracy],
		'Precision': [precision],
		'Recall': [recall],
		'AUC': [auc],
		'Groups Train': [groups_train],
		'Groups Test': [groups_test]
	}

	write_header = not os.path.exists('metric.csv')  # Check if the file exists and set write_header accordingly

	mode = 'w' if write_header else 'a'  # Use 'w' mode to overwrite the file and write headers, 'a' mode to append without headers
	header = True if write_header else False
	df = pd.DataFrame(data)
	df.to_csv('metric.csv', header=header,mode=mode, index=False)

def split_data(X_data, y_data, group_label, fixed_group_to_test):
	#split data into train and test based on the fixed fixed_group_to_test_list
	train_idx = []
	test_idx = []
	for i in range(len(group_label)):
		if group_label[i] in fixed_group_to_test:
			test_idx.append(i)
		else:
			train_idx.append(i)
	X_train, X_test = X_data[train_idx], X_data[test_idx]
	y_train, y_test = y_data[train_idx], y_data[test_idx]
	groups_train, groups_test = group_label[train_idx], group_label[test_idx]
	return(X_train, X_test, y_train, y_test, groups_train, groups_test)