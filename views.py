from __future__ import print_function

import os
import pandas as pd
import datetime

import jinja2
import json

from stanFunction import *
from utils import *
from flask import Flask, flash, request, redirect, url_for, render_template, send_file, jsonify
from werkzeug.utils import secure_filename

# creating an instance of the Flask class
app = Flask(__name__)
UPLOAD_FOLDER = 'tmp/'

##### STAN MODELS
IM = loadModel("inference")
CM = loadModel("comparaison")

##### VIEWS
@app.route("/", methods=["GET", "POST"], endpoint="index")
def main() :
	return render_template("main.html", title="Home")

@app.route("/inference/<form>", methods=["GET", "POST"], endpoint="inference")
def inference(form) :
	return render_template("inference.html", title="Inference", form=form)

@app.route("/comparaison/<form>", methods=["GET", "POST"], endpoint="comparaison")
def comparaison(form) :
	return render_template("comparaison.html", title="Comparaison", form=form, ms=[mu1, sigma1, mu2, sigma2])

@app.route("/ERROR/<template>/<error>", methods=["GET", "POST"], endpoint="error")
def error(template, error) :
	return render_template("%s.html" % (template), title="Error", error=error, form="files")

@app.route("/<template>/plot/<ID>", methods=["GET", "POST"], endpoint="plot")
def plot(template, ID) :
	if template == "comparaison" :
		data = []

		data.append(("Percentile Values - Dataset1", pd.read_pickle("tmp/table_%s_1" % ID)))
		data.append(("Percentile Values - Dataset2", pd.read_pickle("tmp/table_%s_2" % ID)))
		data.append(("Percentile Values - Comparaison", pd.read_pickle("tmp/table_%s_3" % ID)))

	else :
		data = [("Percentile Values", pd.read_pickle("tmp/table_%s" % ID))]
	return render_template("%s.html" % (template), title="Analysis", graph=True, ID=ID, tableData=data, form="files")

@app.route("/display/<drc>/<img>", methods=["GET"], endpoint="display")
def display(drc, img) :
	return send_file(os.path.join(drc, img), mimetype="image/png")



##### ANALYSIS SELECTION
@app.route("/selection", methods=["GET", "POST"], endpoint="selection")
def selection() :
	if request.method == "POST" :
		selected = request.form["selected"]

		return redirect(url_for(selected, form="files", args=[]))

@app.route("/upload", methods=["GET", "POST"], endpoint="upload")
def upload() :
	global ID
	ID = uniqueID()

	global userData
	userData = []

	#Extract data from teh HTML form
	if request.method == "POST" :
		global graphInfo
		graphInfo = request.form

		### Save file(s)
		if request.files.keys() :
			c = 1

			for dataset in request.files.keys() :
				file = request.files[dataset]
				tmpOutput = saveFiles(file, graphInfo["submit"], ID, c)

				if isinstance(tmpOutput, pd.DataFrame) :
					#global userData
					userData.append(tmpOutput)

				else :
					return tmpOutput

				c += 1

		else :
			return redirect(url_for("error", template=graphInfo["submit"], error="Dataset(s) file(s) must be uploaded"))

	return redirect(url_for(graphInfo["submit"], form="priors"))


@app.route("/analyze", methods=["GET", "POST"], endpoint="analyze")
def analyze():
	if request.method == "POST":
		priorInfo = request.form

		analysis = priorInfo["submit"]

		if analysis == "inference" :
			print (userData)
			x, y, x_infer = extractData(userData[0])

			print ('1 - Organizing the data')	
			stanData = {"N" : len(x), "x" : x, "y" : y,
						"N_infer" : len(x_infer), "x_infer" : x_infer,
						"LDR_mu" : float(priorInfo["LDR_mu"]), "LDR_sigma" : float(priorInfo["LDR_sigma"]),
						"HDR_mu" : float(priorInfo["HDR_mu"]), "HDR_sigma" : float(priorInfo["HDR_sigma"]), "HDR_alpha" : float(priorInfo["HDR_alpha"]),
						"I_alpha" : float(priorInfo["I_alpha"]), "I_beta" : float(priorInfo["I_beta"]),
						"S_mu" : float(priorInfo["S_mu"]), "S_sigma" : float(priorInfo["S_sigma"]),
						"s_pos" : float(priorInfo["s_pos"]), "s_scale" : float(priorInfo["s_scale"])}

			print ('2 - Running the model')
			stanResult = runModel(IM, stanData)

			print ('3 - Plotting the results')
			plotInference(ID, x, y, x_infer, graphInfo, stanResult, "", priorInfo)
			pairwiseInference(ID, x, y, x_infer, graphInfo, stanResult, "", priorInfo)
			print ('4 - Creating the results tables')
			tableDataInference(stanResult, "", "", ID, priorInfo)

		elif analysis == "comparaison" :
			x1, y1, x_infer1 = extractData(userData[0])
			x2, y2, x_infer2 = extractData(userData[1])
			x_infer = np.sort(list(set(x_infer1).union(set(x_infer2))))

			stanData = {'N1':len(x1), 'x1':x1, 'y1':y1,  
						'N2':len(x2), 'x2':x2, 'y2':y2,
						'N_inference':len(x_infer), 'x_inference':x_infer,
						"LDRmu1" : float(priorInfo["LDR_mu1"]), "LDRsigma1" : float(priorInfo["LDR_sigma1"]),
						"LDRmu2" : float(priorInfo["LDR_mu2"]), "LDRsigma2" : float(priorInfo["LDR_sigma2"]),
						"HDRmu1" : float(priorInfo["HDR_mu1"]), "HDRsigma1" : float(priorInfo["HDR_sigma1"]),
						"HDRmu2" : float(priorInfo["HDR_mu2"]), "HDRsigma2" : float(priorInfo["HDR_sigma2"]),
						"Imu1" : float(priorInfo["I_mu1"]), "Isigma1" : float(priorInfo["I_sigma1"]),
						"Imu2" : float(priorInfo["I_mu2"]), "Isigma2" : float(priorInfo["I_sigma2"]),
						"Smu1" : float(priorInfo["S_mu1"]), "Ssigma1" : float(priorInfo["S_sigma1"]),
						"Smu2" : float(priorInfo["S_mu2"]), "Ssigma2" : float(priorInfo["S_sigma2"])}

			stanResult = runModel(CM, stanData)

			plotComparaison(ID, [x1, x2], [y1, y2], x_infer, graphInfo, stanResult, priorInfo)
			pairwiseComparaison(ID, [x1, x2], [y1, y2], x_infer, graphInfo, stanResult, priorInfo)
			tableDataComparaison(stanResult, ID, priorInfo)

		return redirect(url_for('plot', template=analysis, ID=ID))

	 

