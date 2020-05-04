import os
from pystan import *
import pickle

dataType = ['activity', 'inhibition']

def loadModel(analysis) :
	models = {}

	for dt in dataType :
		models[dt] = pickle.load(open("stan/%s_%s_model.pkl" % (dt, analysis), "rb"))

	return models

def runModel(sm, data) :
	print ('--- Running the model')
	
	fit = sm.sampling(data=data, chains=4, iter=4000, warmup=2000)
	print (fit)
	
	print ('--- Extracting the results')
	stanResult = fit.extract()

	return stanResult
