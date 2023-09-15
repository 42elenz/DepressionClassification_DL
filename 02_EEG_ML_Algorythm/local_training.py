import training_functions as t
import os
from tensorflow.keras.optimizers import Adam
import keras
import tensorflow as tf
import numpy as np

def make_X_y_group_data(data_dir, sliding_window_size=4, sliding_window_overlap=0.5):
	#load and preprocess the data for Healthy
	group_label = []
	dataDir_H = os.path.join(data_dir,"H_EC")
	group_label, data_H = t.load_data(dataDir=dataDir_H,group_label=group_label, sliding_window_size=sliding_window_size, sliding_window_overlap=sliding_window_overlap, from_array=True)

	#load and preprocess the data for MDD
	dataDir_MDD = os.path.join(data_dir,"MDD_EC")
	group_label, data_MDD = t.load_data(dataDir=dataDir_MDD,group_label=group_label, sliding_window_size=sliding_window_size, sliding_window_overlap=sliding_window_overlap, from_array=True)

	#Trim the shape of the data if necessary
	#data_H, data_MDD= t.control_for_shape(data_H, data_MDD)

	#make X and y data
	X_data,y_data=t.makeX_Y_data(data_H, data_MDD)
	return(X_data, y_data, group_label)

def set_directories(controll_dir):
	dataDir = os.path.join(controll_dir,"train_data/")
	model_path = os.path.join(controll_dir,"ref_model")
	return(dataDir, model_path)

def local_training(controll_dir):
	#set max_epochs
	max_epochs = 30

	#set the directories
	dataDir, model_path = set_directories(controll_dir)

	#make the X_data and y_data
	X_data, y_data, group_label = make_X_y_group_data(dataDir, sliding_window_size=4, sliding_window_overlap=0.5)

	#make X_test and y_test
	X_test, y_test, test_groupp_label = make_X_y_group_data(controll_dir, sliding_window_size=4, sliding_window_overlap=0.5)

	#Permutate and shuffel data
	X_train, y_train, group_label = t.permutate(X_data, y_data, group_label)

	#expand variables
	expanded_X, expanded_y = t.expand_dim([X_train, X_test], [y_train, y_test])
	X_train, X_test = expanded_X
	y_train, y_test = expanded_y
	groups_val=[]

	#set a random state
	np.random.seed(4)
	tf.random.set_seed(4)
	tf.keras.utils.set_random_seed(4)
	#define the model architecture with keras
	architecture_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])
	model = t.model_architecture(architecture_shape)
	opt = Adam( learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adam' )
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()])

	# if len(groups_val) != 0:
	# 	model.fit(X_train, y_train, batch_size=32, epochs=max_epochs, validation_data=(X_val, y_val))
	# else:
	#print randome state
	print('random state: ', np.random.get_state()[1][0])
	model.fit(X_train, y_train, batch_size=32, epochs=max_epochs)

	# evaluate the model
	loss, accuracy, precision, recall, auc = model.evaluate(X_test, y_test)

	#save the model not with swarm
	model.save(model_path)

	#append the metrics and which groups were used in csv file
	filename=os.path.join(controll_dir, 'local_metrics.csv')
	t.write_to_csv([controll_dir, loss, accuracy, precision, recall, auc, '\n'], filename)
	
	#reset random states
	np.random.seed(None)
	tf.random.set_seed(None)

	print("Process two local_training finished")
	return 0
