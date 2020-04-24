import os
import pickle
import pystan

def saveModel(analysis) :
	print ("----- %s" % (analysis))
	print ("-------- Compiling")
	sm = pystan.StanModel(file="%sModel.stan" % (analysis))

	print ("-------- Saving")
	with open("%sModel.pkl" % (analysis), "wb") as f :
		pickle.dump(sm, f)

	print ("-------- Done")


saveModel("inference")
saveModel("comparaison")
