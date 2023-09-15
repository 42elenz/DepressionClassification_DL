import training as t
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
import csv
import numpy as np
import os
import random as rn
from sklearn.metrics import roc_curve

from numpy.random import seed

def run_iteration(window_size, window_overlap, random_state, fixed_group_to_test, accuracy):

	#Load and preprocess the data for MDD
	seed(random_state)
	tf.random.set_seed(random_state)
	os.environ['PYTHONHASHSEED']=str(random_state)
	rn.seed(random_state)
	tf.keras.utils.set_random_seed(42)

	group_label=[]
	data_dir = 'data/MDD_EC'
	group_label, data_MDD = t.load_data(dataDir=data_dir,group_label=group_label, sliding_window_size=window_size, sliding_window_overlap=window_overlap)
	t.print_data_tofile(data_MDD)

	#Load and preprocess the data for Healthy
	data_dir = 'data/H_EC'
	group_label, data_H = t.load_data(dataDir=data_dir,group_label=group_label, sliding_window_size=window_size, sliding_window_overlap=window_overlap)
	t.print_data_tofile(data_H)

	#Trim the shape of the data if necessary
	#data_H, data_MDD= t.control_for_shape(data_H, data_MDD)

	#make X and y data
	X_data,y_data=t.makeX_Y_data(data_H, data_MDD)

	#Permutate and shuffel data
	X_data, y_data, group_label = t.permutate(X_data, y_data, group_label,random_state)

	#Split data into train and test
	X_train, X_test, y_train, y_test, groups_train, groups_test = t.split_data(X_data, y_data, group_label, fixed_group_to_test)

	#Stratified Group K Fold
	#groups_train, groups_val, groups_test, X_train, X_val, X_test, y_train, y_val, y_test = t.stratified_group_k_fold(X_data, y_data, group_label)

	#expand variables
	expanded_X, expanded_y = t.expand_dim([X_train, X_test], [y_train, y_test])
	X_train, X_test = expanded_X
	y_train, y_test = expanded_y

	print('######################################')
	print(X_train.shape)
	print(X_test.shape)
	print(y_train.shape) 
	print(y_test.shape)

	#define model
	architecture_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])
	model = t.model_architecture(architecture_shape)
	opt = Adam( lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adam' )
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()])

	#train model
	model.fit(X_train, y_train, batch_size=32, epochs=30)
	loss, model_accuracy, precision, recall, auc = model.evaluate(X_test, y_test)
	if model_accuracy > accuracy:
		#Save the model
		model.save('ref_model')
	#write the metrcis in a csv file
	t.add_metrics_to_csv(random_state,window_size,window_overlap,loss, model_accuracy, precision, recall, auc, np.unique(groups_train), np.unique(groups_test))
	return(model_accuracy)
def main():
	#define window parameter for loop
	accuracy = 0
	random_state_list = [4]
	sliding_window_size = [16]
	sliding_window_overlap = [0.8]
	fixed_group_to_test = [5,6,7,8,37,48,39,40]
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
	# Limit memory growth for each GPU
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
	for random_state in random_state_list:
		for window_size in sliding_window_size:
			for window_overlap in sliding_window_overlap:
				#tf.keras.backend.clear_session()
				run_accuracy = run_iteration(window_size, window_overlap, random_state, fixed_group_to_test, accuracy)
				if run_accuracy > accuracy:
					accuracy = run_accuracy

if __name__=='__main__':
	main()
