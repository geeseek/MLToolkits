from math import exp
from math import log 
import numpy as np

def sigmoid(z):
	return 1.0/(1 + exp(-1 * z))

def predict(theta, x):
	h = sigmoid(np.dot(theta.T, x))
	return h

def calcError(theta, X, y):
	m = y.size
	count = 0
	error = 0
	while count < m:
		h = sigmoid(np.dot(theta.T, X[count]))	
		c = 0
		if h > 0.5:
			c = 1
		if abs(c - y[count]) > 0.1:
			error = error + 1
		count = count + 1
	return error * 1.0 / m;
	

def costFunction(theta, X, y, lamb):
	m = y.size 
	count = 0
	J = 0
	while count < m:
		h = sigmoid(np.dot(theta.T, X[count]))
		J += -y[count]*log(h) - (1 - y[count])*(log(1.00001 - h)) 
		count += 1
	J = J / m
	J = J + np.dot(theta.T, theta) * lamb / (2 * m)
	return J

def updateTheta(theta, X, y, eta, lamb):
	'''compute grad'''
	m = y.size
	grad = np.zeros(theta.size)
	dim = 0
	while dim < grad.size:
		num = 0
		while num < m:
			h = sigmoid(np.dot(theta.T, X[num]))
			grad[dim] += (h - y[num]) * X[num][dim] + theta[dim] * lamb
			num = num + 1
		grad[dim] = grad[dim] / m
		dim = dim + 1

	'''update theta'''
	i = 0
	while i < theta.size:
		theta[i] = theta[i] - grad[i] * eta
		i = i + 1
	return theta

def loadData(path):
	fileHandle = open(path)
	records = fileHandle.readlines()
	numOfSample = len(records)
	numOfFeatures = len(records[0].split(','))
	X = np.zeros((numOfSample, numOfFeatures))
	y = np.zeros(numOfSample)
	i = 0 
	while i < numOfSample:
		fields = records[i].split(',')
		j = 0
		X[i][j] = 1
		while j < len(fields) - 1:
			X[i][j+1] = fields[j]
			j += 1
		y[i] = fields[len(fields) - 1]
		i += 1
	return (X, y)

'''train data:  https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw3%2Fhw3_train.dat '''
'''test data: https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw3%2Fhw3_test.dat'''

path = raw_input('pls input the training file name:\n')
X,y = loadData(path) 
theta = np.zeros((len(X[0]),1))
iterNum = 2000
lamb = 0.01
eta = 0.05
count = 0
lastJ = 0
while count < iterNum:
	lastJ = costFunction(theta, X, y, lamb)
	theta = updateTheta(theta, X, y, eta, lamb)
	J = costFunction(theta, X, y, lamb)
	print "J=" + str(J)
	if abs(J-lastJ) < 0.00001:
		print "stop at round " + str(count)
		break;
	count = count + 1

print "E_in=" + str(calcError(theta, X, y))
tpath = raw_input('pls input the test file name:\n')
tX,ty = loadData(tpath)
print theta
print "E_out=" + str(calcError(theta, tX, ty))
