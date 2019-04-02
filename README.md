# NeuralNetworkGenerator
This library provides full process to Test and generate the best Neural Net model based on a series of parameters 
for a generic input dataframe 

### load_processed_data
- load data  
### generate_nn_list
- Return a list of dictionnaries. Each dict has different param for a neural net. The input is a dict of parameters : 
	param_choices = {
	'nb_neurons' : [32,64,128],
	'nb_layers' : [3,4,5,6],
	'activation' : ['relu','elu','sigmoid'],
	'optimizer' : ['adam','rmsprop','adamax','sgd'],
	}
### compile_model
- Build a neural net from generate_nn_list output
### train_networks
- Classic Train and test models
### cross_validation
- Train and Test models on n differents train_test_split of the data, and return the score mean for each architecture of neural net
