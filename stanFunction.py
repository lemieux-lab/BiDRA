import os
from pystan import *
import pickle

def loadModel(analysis) :
	return pickle.load(open("stan/%sModel.pkl" % (analysis.lower()), "rb"))

def runModel(sm, data) :
	print ('--- Running the model')
	print (data)
	
	fit = sm.sampling(data=data, chains=4, iter=4000, warmup=2000)
	print (fit)
	
	print ('--- Extracting the results')
	stanResult = fit.extract()

	return stanResult
