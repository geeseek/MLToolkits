from math import exp
from math import log 
import numpy as np
from numpy import *

def sigmoid(z):
	return 1.0/(1 + exp(-1 * z))

def predict(theta, x):
	h = sigmoid(np.dot(theta.T, x))
	return h

def costFunction(data, maxUserId, maxItemId, theta, x):
	J = 0
	i = 0
	while i <= maxUserId:
		j = 0
		while j <= maxItemId:
			if (data[i][j] > 0):
				J += pow(np.dot(theta[i], (x[j]).T) - data[i][j], 2)
			j = j + 1
		i = i + 1
	J /= 2
	'''
	reguTheta = 0
	i = 0
	while i <= maxUserId:
		reguTheta += sum(theta[i]**2)
		i += 1
	reguX = 0
	j = 0
	while j <= maxItemId:
		reguX += sum(x[j]**2)
		j += 1
	'''
	return J

def updateTheta(data, maxUserId, maxItemId, theta, x, eta, lamb):
	'''compute grad'''
	grad = np.zeros(theta.shape)
	dimOfFactors = grad.shape[1]
	k = 0
	while k < dimOfFactors:
		i = 0
		while i <= maxUserId:
			j = 0
			while j <= maxItemId:
				if data[i][j] > 0:
					h = np.dot(theta[i], (x[j]).T)
					grad[i][k] += (h - data[i][j]) * x[j][k]
				j += 1
			i += 1
		k += 1	

	'''update theta'''
	i = 0
	while i <= maxUserId:
		theta[i] = theta[i] - grad[i] * eta
		i = i + 1
	return theta

def updateX(data, maxUserId, maxItemId, theta, x, eta, lamb):
	'''compute grad'''
	grad = np.zeros(x.shape)
	dimOfFactors = grad.shape[1]
	j = 0
	while j <= maxItemId:
		i = 0
		while i <= maxUserId:
			if data[i][j] > 0:
				h = np.dot(theta[i], (x[j]).T)
				k = 0
				while k < dimOfFactors:
					grad[j][k] += (h - data[i][j]) * theta[i][k]
					k += 1
			i += 1
		j += 1
	'''update theta'''
	j = 0
	while j <= maxItemId:
		x[j] = x[j] - grad[j] * eta
		j = j + 1
	return x 


def loadData(path):
	fileHandle = open(path)
	records = fileHandle.readlines()
	numOfRatings = len(records)
	numOfUsers = 1000 
	numOfItems = 10000 
	data = np.zeros((numOfUsers, numOfItems))
	y = np.zeros(numOfRatings)
	i = 0 
	maxUserId = 0
	maxItemId = 0
	while i < numOfRatings:
		fields = records[i].split('\t')
		userId = int(fields[0]) - 1
		itemId = int(fields[1]) - 1
		rating = int(fields[2])
		data[userId][itemId] = rating 
		if userId > maxUserId:
			maxUserId = userId
		if itemId > maxItemId:
			maxItemId = itemId
		i += 1
	return (data, maxUserId, maxItemId)

def train(data, maxUserId, maxItemId, eta, lamb, maxIter):
	numOfFactors = 5 
	theta = np.random.rand(maxUserId + 1, numOfFactors)
	x = np.random.rand(maxItemId + 1, numOfFactors)
	iter = 0
	while iter < maxIter:
		J = costFunction(data, maxUserId, maxItemId, theta, x)
		print "round " + str(iter) + ":" + " J=" + str(J)
		theta = updateTheta(data, maxUserId, maxItemId, theta, x, eta, lamb)
		x = updateX(data, maxUserId, maxItemId, theta, x, eta, lamb)
		iter += 1
	return theta, x


'''tested with movielens 100k dataset'''
path = raw_input("pls input the path of train file\n")
data, maxUserId, maxItemId = loadData(path)
maxIter = int(raw_input("pls input the number of iteration\n"));
print maxIter
eta = 0.001
lamb = 0.1
theta, x = train(data, maxUserId, maxItemId, eta, lamb, maxIter)
np.savez('model', theta=theta, x = x)

print "E_in = " + str(costFunction(data, maxUserId, maxItemId, theta, x))

tpath = raw_input("pls input the path of test file\n")
data, maxUserId, maxItemId = loadData(tpath)

'''
model = np.load('model.npz')
theta = model['theta']
x = model['x']
'''

print "E_out = " + str(costFunction(data, maxUserId, maxItemId, theta, x))
