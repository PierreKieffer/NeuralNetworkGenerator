'''

Train and test multiple neural net architectures on the same dataset, to find the best.

- load_processed_data : load data  
- generate_nn_list : Return a list of dictionnaries. Each dict has different param for a neural net. The input is a dict of parameters : 

	param_choices = {
	'nb_neurons' : [32,64,128],
	'nb_layers' : [3,4,5,6],
	'activation' : ['relu','elu','sigmoid'],
	'optimizer' : ['adam','rmsprop','adamax','sgd'],
	}

- compile_model : Build a neural net from generate_nn_list output
- train_networks : Classic Train and test models
- cross_validation : Train and Test models on n differents train_test_split of the data, and return the score mean for each architecture of neural net

'''

###################
import pandas as pd 
import numpy as np 
import argparse
import sys 

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

from sklearn.model_selection import train_test_split 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

###################


def load_processed_data(path, label_column): 

	data = pd.read_csv(path, index_col = 0)

	input_size = len(data.columns) -1

	# train test split 
	features = data.drop([label_column], axis = 1)
	labels = data[label_column]

	labels = np_utils.to_categorical(labels)
	nb_classes = labels.shape[1]
	

	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2)

	return X_train, X_test, y_train, y_test, nb_classes, input_size


def generate_nn_list(param_choices):
	'''
	Return list of dict with differents param 

	Arguments : 
	- param_choices = dict with lists of parametres

	'''

	networks = []

	for nbn in param_choices['nb_neurons']:
		for nbl in param_choices['nb_layers']:
			for a in param_choices['activation']:
				for o in param_choices['optimizer']:

					network = {
					'nb_neurons' : nbn,
					'nb_layers' : nbl,
					'activation' : a,
					'optimizer' : o,
					}

					networks.append(network)

	return networks


def compile_model(network, nb_classes, input_size):
	'''
	Compile MLP based on generate_nn_list

	Arguments : 
	- network : dict with network params 
	- nb_classes : int from data load
	- input_size : int from data load
	'''

	# Get parameters 
	nb_layers = network['nb_layers']
	nb_neurons = network['nb_neurons']
	activation = network['activation']
	optimizer = network['optimizer']

	model = Sequential()

	# Add each layer 
	for i in range(nb_layers):

		# Add input shape for first layer 
		if i == 0:
			model.add(Dense(nb_neurons, activation = activation, input_shape = (input_size,) ))
		else :
			model.add(Dense(nb_neurons, activation = activation))
		model.add(Dropout(0.2))

		# Add output layer 
		model.add(Dense(nb_classes, activation = 'softmax'))

		model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

		return model 


def train_networks(networks, X_train, X_test, y_train, y_test, nb_classes, input_size, batch_size, nb_epochs):
	'''
	Train each network 
	Return dataframe with perf of each network

	Arguments : 
	- networks = list of dicts (from generate_nn_list)
	- X_train, X_test, y_train, y_test from train test split in data load
	'''
	# Create performance dataframe 
	perf_dataframe = pd.DataFrame(columns = ['nb_neurons', 'nb_layers','activation','optimizer','score'])


	for network in networks:

		model = compile_model(network, nb_classes, input_size)

		# Log info 
		print(60*"--")
		print(60 * '--')
		print( "Architecture : -- nb_layers = {} -- nb_neurons = {} -- activation = {} -- optimizer = {} --".format(network['nb_layers'], network['nb_neurons'], network['activation'], network['optimizer']))

		model.fit(X_train, y_train, batch_size = batch_size, epochs = nb_epochs, validation_data = (X_test, y_test))
		score = model.evaluate(X_test, y_test, verbose = 0)
		score = score[1]*100

		# Save to dataframe
		temp = pd.DataFrame({'nb_neurons' : network['nb_neurons'], 'nb_layers' : network['nb_layers'],
			'activation' : network['activation'], 'optimizer' : network['optimizer'], 'score' : score}, index = [0])

		perf_dataframe = perf_dataframe.append(temp, ignore_index = True)

	return perf_dataframe


def cross_validation(networks, batch_size, nb_epochs, n_splits):

	# performance dataframe
	perf_dataframe = pd.DataFrame(columns = ['i','nb_neurons', 'nb_layers','activation','optimizer','score'])

	for i in range(n_splits):
		print(10*'//')
		print('split {}'.format(i))

		X_train, X_test, y_train, y_test, nb_classes, input_size = load_processed_data('data.csv', 'label')

		for network in networks:

			print(60*"--")
			print(60 * '--')
			print( "Architecture : -- nb_layers = {} -- nb_neurons = {} -- activation = {} -- optimizer = {} --".format(network['nb_layers'], network['nb_neurons'],network['activation'], network['optimizer']))

			model = compile_model(network, nb_classes, input_size)
			model.fit(X_train, y_train, batch_size = batch_size, epochs = nb_epochs, validation_data = (X_test, y_test))
			score = model.evaluate(X_test, y_test, verbose = 0)
			score = score[1]*100

			temp = pd.DataFrame({'dataset' : i, 'nb_neurons' : network['nb_neurons'], 'nb_layers' : network['nb_layers'],'activation' : network['activation'], 'optimizer' : network['optimizer'], 'score' : score}, index = [0])

			perf_dataframe = perf_dataframe.append(temp, ignore_index = True)

	print(perf_dataframe)
	perf_dataframe = pd.DataFrame({'ScoreMean' : perf_dataframe.groupby(['nb_neurons','nb_layers', 'activation','optimizer'])['score'].mean()}).reset_index()
	print(perf_dataframe)


def main(): 

	# Load data 
	#X_train, X_test, y_train, y_test, nb_classes, input_size = load_processed_data('data.csv', 'label')


	param_choices = {
	'nb_neurons' : [32,64],
	'nb_layers' : [2,4],
	'activation' : ['relu','elu'],
	'optimizer' : ['adam','adamax','rmsprop'],
	}
	
	networks = generate_nn_list(param_choices)

	cross_validation(networks, 100,8,2)


if __name__ == '__main__': 
	main()

	

