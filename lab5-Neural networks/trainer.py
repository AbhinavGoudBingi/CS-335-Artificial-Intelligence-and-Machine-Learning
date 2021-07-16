'''File contains the trainer class

Complete the functions train() which will train the network given the dataset and hyperparams, and the function __init__ to set your network topology for each dataset
'''
import numpy as np
import sys
import pickle

import nn

from util import *
from layers import *

class Trainer:
	def __init__(self,dataset_name):
		self.save_model = False
		if dataset_name == 'MNIST':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readMNIST()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 10
			self.epochs = 10
			self.lr = 0.01
			self.nn = nn.NeuralNetwork(10, self.lr)
			self.nn.addLayer(FullyConnectedLayer(self.XTrain.shape[1],100,'relu'))
			self.nn.addLayer(FullyConnectedLayer(100,10,'softmax'))



		if dataset_name == 'CIFAR10':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readCIFAR10()
			self.XTrain = self.XTrain[0:5000,:,:,:]
			self.XVal = self.XVal[0:1000,:,:,:]
			self.XTest = self.XTest[0:1000,:,:,:]
			self.YVal = self.YVal[0:1000,:]
			self.YTest = self.YTest[0:1000,:]
			self.YTrain = self.YTrain[0:5000,:]

			self.save_model = True
			self.model_name = "model.npy"

			# Add your network topology along with other hyperparameters here
			self.batch_size = 10
			self.epochs = 10
			self.lr = 0.1
			self.nn = nn.NeuralNetwork(10, self.lr)
			self.nn.addLayer(ConvolutionLayer((3,32,32),(4,4),32,2,'relu'))
			output_size = (self.nn.layers[-1].out_depth,self.nn.layers[-1].out_row,self.nn.layers[-1].out_col)
			self.nn.addLayer(MaxPoolingLayer(output_size,(3,3),3))
			output_size = self.nn.layers[-1].out_depth*self.nn.layers[-1].out_row*self.nn.layers[-1].out_col
			self.nn.addLayer(FlattenLayer())
			self.nn.addLayer(FullyConnectedLayer(output_size,45,'relu'))
			self.nn.addLayer(FullyConnectedLayer(45,10,'softmax'))

		if dataset_name == 'XOR':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readXOR()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 10
			self.epochs = 15
			self.lr = 0.1
			self.nn = nn.NeuralNetwork(2, self.lr)
			self.nn.addLayer(FullyConnectedLayer(self.XTrain.shape[1],4,'softmax'))
			self.nn.addLayer(FullyConnectedLayer(4,2,'softmax'))


		if dataset_name == 'circle':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readCircle()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 10
			self.epochs = 15
			self.lr = 0.1
			self.nn = nn.NeuralNetwork(2, self.lr)
			self.nn.addLayer(FullyConnectedLayer(self.XTrain.shape[1],3,'softmax'))
			self.nn.addLayer(FullyConnectedLayer(3,2,'softmax'))         
	def train(self, verbose=True):
		# Method for training the Neural Network
		# Input
		# trainX - A list of training input data to the neural network
		# trainY - Corresponding list of training data labels
		# validX - A list of validation input data to the neural network
		# validY - Corresponding list of validation data labels
		# printTrainStats - Print training loss and accuracy for each epoch
		# printValStats - Prints validation set accuracy after each epoch of training
		# saveModel - True -> Saves model in "modelName" file after each epoch of training
		# loadModel - True -> Loads model from "modelName" file before training
		# modelName - Name of the model from which the funtion loads and/or saves the neural net
		
		# The methods trains the weights and baises using the training data(trainX, trainY)
		# and evaluates the validation set accuracy after each epoch of training

		for epoch in range(self.epochs):
			# A Training Epoch
			if verbose:
				print("Epoch: ", epoch)

			# TODO
			trainX = self.XTrain.copy()
			trainY = self.YTrain.copy()
			# Shuffle the training data for the current epoch
			indices = np.arange(trainX.shape[0])
			indices = np.random.permutation(indices)
			trainX = trainX[indices]
			trainY = trainY[indices]

			# Initializing training loss and accuracy
			trainLoss = 0
			trainAcc = 0

			# Divide the training data into mini-batches

				# Calculate the activations after the feedforward pass
				# Compute the loss  
				# Calculate the training accuracy for the current batch
				# Backpropagation Pass to adjust weights and biases of the neural network
			#trainX = trainX.reshape(-1,1,28,28)
			numBatches = trainX.shape[0]//self.batch_size
			for i in range(numBatches):
				activations = self.nn.feedforward(trainX[i*(self.batch_size):(i+1)*(self.batch_size)])
				trainLoss += self.nn.computeLoss(trainY[i*(self.batch_size):(i+1)*(self.batch_size)],activations)
				a_copy = activations[-1].copy()
				b_copy = np.max(a_copy,axis=1,keepdims=True)
				trainAcc += self.nn.computeAccuracy(trainY[i*(self.batch_size):(i+1)*(self.batch_size)],a_copy==b_copy)
				self.nn.backpropagate(activations,trainY[i*(self.batch_size):(i+1)*(self.batch_size)])

			# END TODO
			# Print Training loss and accuracy statistics
			trainAcc /= numBatches
			trainLoss /= numBatches
			if verbose:
				print("Epoch ", epoch, " Training Loss=", trainLoss, " Training Accuracy=", trainAcc)
			
			if self.save_model:
				model = []
				for l in self.nn.layers:
					# print(type(l).__name__)
					if type(l).__name__ != "AvgPoolingLayer" and type(l).__name__ != "FlattenLayer" and type(l).__name__ != "MaxPoolingLayer": 
						model.append(l.weights) 
						model.append(l.biases)
				pickle.dump(model,open(self.model_name,"wb"))
				print("Model Saved... ")

			# Estimate the prediction accuracy over validation data set
			if self.XVal is not None and self.YVal is not None and verbose:
				_, validAcc = self.nn.validate(self.XVal, self.YVal)
				print("Validation Set Accuracy: ", validAcc, "%")

		pred, acc = self.nn.validate(self.XTest, self.YTest)
		print('Test Accuracy ',acc)

