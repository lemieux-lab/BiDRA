import os
import pickle
import pystan

def saveModel(analysis, data) :
	print ("----- %s (%s)" % (analysis, data))
	print ("-------- Compiling")
	sm = pystan.StanModel(file="%s_%s_model.stan" % (data, analysis))

	print ("-------- Saving")
	with open("%s_%s_model.pkl" % (data, analysis), "wb") as f :
		pickle.dump(sm, f)

	print ("-------- Done")


analysisType = ['inference', 'comparison']
dataType = ['activity', 'inhibition']

for at in analysisType :
	for dt in dataType :

		saveModel(at, dt)
