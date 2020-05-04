import os
import numpy as np
import pandas as pd
import collections
import pickle
import pystan
import scipy.stats as ss 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from random import randint
from flask import Flask, flash, request, redirect, url_for, render_template, send_file, jsonify
from werkzeug.utils import secure_filename

from views import *

# where the files are TEMPORALY stored
UPLOAD_FOLDER = "tmp/"
ALLOWED_EXTENSIONS = set(["csv"])

plt.style.use(["seaborn-whitegrid"])
plt.rc("font", size=12)

param = ["HDR", "LDR", "I", "S", "sigma"]
#plotLabel = ["Maximal Response", "Minimal Response", r"IC$_{50}$", "Steepness (slope)"]
htmlLabel = ["Maximal Response", "Minimal Response", "IC50", "Steepness (slope)"]

def allowed_file(filename) :
	return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def saveFiles(file, template, ID, idx) :
	if file and allowed_file(file.filename) :
		filename = "%s_%s_%s" % (ID, idx, secure_filename(file.filename))
		file.save(os.path.join(UPLOAD_FOLDER, filename))

		return pd.read_csv(os.path.join(UPLOAD_FOLDER, filename), header=None, names=["x", "y"])

	else :
		return redirect(url_for("error", template=template, error="The uploaded file(s) must be CSV"))


def uniqueID():
	now = datetime.datetime.now()
	today = datetime.date.today()
	r = randint(1000, 5000) 
	unique = "%s%s%s" % (now.microsecond, today.toordinal(), r)
	return unique

def getPercentile(data, p) :
	x = data[0]
	y = data[1].T
	
	y_p = []
	
	for i in range(0, len(x)) :
		y_p.append(np.percentile(np.sort(y[i]), p))
	
	return y_p

def norm(x, mu, sigma) :
	return ss.norm.pdf(x, mu, sigma)

def uni(x, alpha, beta) :
	pdf = []

	for i in x :
		if i < alpha or i > beta :
			pdf += [0.0]
		else : 
			pdf += [1.0 / (beta - alpha)]
	
	return pdf

def lognorm(x, mu, sigma) :
	return (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))

def cauchy(x, x0, gamma) :
	return ss.cauchy.pdf(x, x0, gamma)

def extractData(data) :
	x = list(data.iloc[:,0])
	y = list(data.iloc[:,1])

	x_infer = np.arange((np.min(x) - np.abs((4*np.min(x)))), (np.max(x) + np.abs((2*np.max(x)))), 0.1)

	return x, y, x_infer

def extractTableData(data, percentile) :
	tmp = []

	for i in percentile :
		tmp.append(str(np.round(np.percentile(data, i), 4)))

	return tmp

def tableDataInference(data, idx, diff, ID, priorInfo) :
	htmlLabel = [priorInfo["HDR"], priorInfo["LDR"], priorInfo["I"], priorInfo["S"], priorInfo["sigma"]]

	percentile = [0.5, 2.5, 5, 50, 95, 97.5, 99.5]
	df = pd.DataFrame(index=percentile)

	for i in range(0, len(param)-1) :
		df.loc[:, htmlLabel[i]] = extractTableData(data["%s%s%s" % (diff, param[i], idx) ], percentile)

	if idx :
		df.to_pickle("tmp/table_%s_%s" % (ID, idx))
	elif diff :
		df.to_pickle("tmp/table_%s_3" % (ID))
	else :
		df.to_pickle("tmp/table_%s" % (ID))

def tableDataComparison(data, ID, priorInfo) :
	for i in [1, 2] :
		tableDataInference(data, i, "", ID, priorInfo)

	tableDataInference(data, "", "diff", ID, priorInfo)


def plotInference(ID, x, y, x_infer, graphInfo, stanResult, idx, priorInfo) :
	plotLabel = [priorInfo["HDR"], priorInfo["LDR"], priorInfo["I"], priorInfo["S"], priorInfo["sigma"]]

	priors = collections.OrderedDict({
									"LDR%s" % (idx) : [norm, float(priorInfo["LDR_mu%s" % (idx)]), float(priorInfo["LDR_sigma%s" % (idx)])], 
									"HDR%s" % (idx) : [norm, float(priorInfo["HDR_mu%s" % (idx)]), float(priorInfo["HDR_sigma%s" % (idx)])], 
									"I%s" % (idx) : [uni, float(priorInfo["I_alpha%s" % (idx)]), float(priorInfo["I_beta%s" % (idx)])], 
									"S%s" % (idx) : [lognorm, float(priorInfo["S_mu%s" % (idx)]), float(priorInfo["S_sigma%s" % (idx)])],
									"sigma%s" % (idx) : [cauchy, float(priorInfo["s_pos%s" % (idx)]), float(priorInfo["s_scale%s" % (idx)])]})

	if idx :
		C = "C%s" % (idx-1)
	else :
		C = "C1"

	fig, ax = plt.subplots(1, 6, sharex="col", figsize=(24, 4))

	### Curve subplot
	ax[0].axhline(y=np.median(stanResult["HDR%s" % (idx)]), ls="-", color="k", lw=1)
	ax[0].axhline(y=np.median(stanResult["LDR%s" % (idx)]), ls="-", color="k", lw=1)
	ax[0].axvline(x=np.median(stanResult["I%s" % (idx)]), ls="-", color="k", lw=1)

	ax[0].plot(x, y, "o", color=C)
	ax[0].axvspan(np.min(x), np.max(x), alpha=0.3, color="grey")

	ax[0].plot(x_infer, getPercentile([x_infer, stanResult["y_infer%s" % (idx)]], 50), "-", color=C, lw=2)
	ax[0].plot(x_infer, getPercentile([x_infer, stanResult["y_infer%s" % (idx)]], 2.5), "--", color=C, lw=1)
	ax[0].plot(x_infer, getPercentile([x_infer, stanResult["y_infer%s" % (idx)]], 97.5), "--", color=C, lw=1)

	ax[0].set_xlabel(graphInfo["xLabel"])
	ax[0].set_ylabel(graphInfo["yLabel"])
	ax[0].set_title("Median curve with\n95%s confidence interval" % ("%"))


	### Parameter estimations
	for k in range(0, len(param)) :
		ax[k+1].hist(stanResult["%s%s" % (param[k], idx)], density=True, color="k", alpha=0.7)

		ax[k+1].axvline(np.median(stanResult["%s%s" % (param[k], idx)]), ls="-", label=str(np.round(np.median(stanResult["%s%s" % (param[k], idx)]), 2)), color=C)
		ax[k+1].axvline(np.percentile(stanResult["%s%s" % (param[k], idx)], 2.5), ls="--", color=C)
		ax[k+1].axvline(np.percentile(stanResult["%s%s" % (param[k], idx)], 97.5), ls="--", color=C)

		dist = priors["%s%s" % (param[k], idx)][0]

		xlim = (ax[k+1].get_xlim())
		xPrior = np.arange(xlim[0], xlim[1], 0.1)
		yPrior = dist(xPrior, priors["%s%s" % (param[k], idx)][1], priors["%s%s" % (param[k], idx)][2])
		ax[k+1].plot(xPrior, yPrior, "-", color="grey", lw=2, alpha=0.7, label="Prior")

		ax[k+1].set_xlabel("Values")
		ax[k+1].set_title("%s\n(95%s c.i.)" % (plotLabel[k], "%"))
		ax[k+1].legend()

	ax[1].set_ylabel("Density")

	if idx :
		plt.suptitle(graphInfo["labelDataset%s" % (idx)], fontsize=20)
	else :
		plt.suptitle(graphInfo["title"], fontsize=20)

	plt.tight_layout(rect=[0, 0.03, 1, 0.9])

	if idx :
		plt.savefig("tmp/plot_%s_%s.pdf" % (ID, idx), orientation="landscape", format="pdf")
		plt.savefig("tmp/plot_%s_%s.png" % (ID, idx), orientation="landscape", format="png")
	else :
		plt.savefig("tmp/plot_%s.pdf" % (ID), orientation="landscape", format="pdf")
		plt.savefig("tmp/plot_%s.png" % (ID), orientation="landscape", format="png")

def pairwiseInference(ID, x, y, x_infer, graphInfo, stanResult, idx, priorInfo):
	plotLabel = [priorInfo["HDR"], priorInfo["LDR"], priorInfo["I"], priorInfo["S"]]

	if idx :
		C = "C%s" % (idx-1)
	else :
		C = "C1"

	fig, ax = plt.subplots(3, 3, figsize=(6, 6), sharex='col', sharey='row')

	for i in range(0, len(param)-2) :
		paramData1 = stanResult["%s%s" % (param[i], idx)]

		for j in range(i, len(param)-2) :
			paramData2 = stanResult["%s%s" % (param[j+1], idx)]

			ax[j]
			ax[j][i].plot(paramData1, paramData2, 'o', c=C, alpha=0.7)
			ax[j][0].set_ylabel(plotLabel[j+1])

		ax[2][i].set_xlabel(plotLabel[i])

	ax[0][1].axis('off')
	ax[0][2].axis('off')
	ax[1][2].axis('off')

	if idx :
		plt.suptitle(graphInfo["labelDataset%s" % (idx)], fontsize=20)
	else :
		plt.suptitle(graphInfo["title"], fontsize=20)

	plt.tight_layout(rect=[0, 0.03, 1, 0.9])

	if idx :
		plt.savefig("tmp/pairwise_%s_%s.pdf" % (ID, idx), orientation="landscape", format="pdf")
		plt.savefig("tmp/pairwise_%s_%s.png" % (ID, idx), orientation="landscape", format="png")
	else :
		plt.savefig("tmp/pairwise_%s.pdf" % (ID), orientation="landscape", format="pdf")
		plt.savefig("tmp/pairwise_%s.png" % (ID), orientation="landscape", format="png")	
	

def plotComparison(ID, x, y, x_infer, graphInfo, stanResult, priorInfo) :
	plotLabel = [priorInfo["HDR"], priorInfo["LDR"], priorInfo["I"], priorInfo["S"]]

	plotInference(ID, x[0], y[0], x_infer, graphInfo, stanResult, 1, priorInfo)
	plotInference(ID, x[1], y[1], x_infer, graphInfo, stanResult, 2, priorInfo)

	fig, ax = plt.subplots(2, 5, figsize=(19, 8))

	### Curves and Data
	for i in [1, 2] :
		l = getPercentile([x_infer, stanResult["y_infer%s" % (i)]], 2.5)
		u = getPercentile([x_infer, stanResult["y_infer%s" % (i)]], 97.5)

		ax[0][0].plot(x_infer, l, "--", lw=1, color="C%d" % (i-1))
		ax[0][0].plot(x_infer, u, "--", lw=1, color="C%d" % (i-1))
		ax[0][0].fill_between(x_infer, l, u, color="C%d" % (i-1), alpha = 0.3)

		ax[0][0].plot(x[i-1], y[i-1], "o", label=graphInfo["labelDataset%s" % (i)])
		ax[0][0].plot(x_infer, getPercentile([x_infer, stanResult["y_infer%s" % (i)]], 50), "-", lw=2, color="C%d" % (i-1))

	ax[0][0].set_xlabel(graphInfo["xLabel"])
	ax[0][0].set_ylabel(graphInfo["yLabel"])
	ax[0][0].set_title("Median curves with\n95%s confidence intervals" % ("%"))
	
	ax[0][0].legend()

	### Parameter and difference distributions
	for k in range(0, len(param)-1) :
		paramData = [stanResult["%s1" % (param[k])], stanResult["%s2" % (param[k])]]

		ax[0][k+1].hist(paramData, stacked=True, color=["C0", "C1"], density=True)

		median1 = np.median(stanResult["%s1" % (param[k])])
		median2 = np.median(stanResult["%s2" % (param[k])])

		ax[0][k+1].axvline(x=median1, ls="-", color="k", label="%s\n(%s)" % (graphInfo["labelDataset1"], str(np.round(median1, 2))))
		ax[0][k+1].axvline(x=median2, ls="--", color="k", label="%s\n(%s)" % (graphInfo["labelDataset2"], str(np.round(median2, 2))))

		ax[0][k+1].set_title(plotLabel[k])
		ax[0][k+1].set_xlabel("Values")
		ax[0][k+1].legend(prop={"size": 10})

		ax[1][k+1].hist(stanResult["diff%s" % param[k]], color="k", alpha=0.7, density=True)
		ax[1][k+1].axvline(x=0.0, ls="-", color="r")

		locX = ax[1][k+1].get_xticks()
		locY = ax[1][k+1].get_yticks()

		diffX = (locX[1] - locX[0]) / 2
		diffY = locY[1] - locY[0]

		ax[1][k+1].set_xlim(locX[0]-diffX, locX[len(locX)-1]+diffX)

		n = float(len(stanResult["diff%s" % param[k]]))
		larger = float(len(stanResult["diff%s" % param[k]][stanResult["diff%s" % param[k]] > 0.0]))
		dt1 = str(np.round((1 - larger/n) * 100, 2)) + "%"
		dt2 = str(np.round((larger/n) * 100, 2)) + "%"

		ax[1][k+1].text(x=locX[0]-diffX, y=locY[len(locY)-3], s=dt1, fontsize=15, color="C0", horizontalalignment="left")
		ax[1][k+1].text(x=locX[len(locX)-1]+diffX, y=locY[len(locY)-3], s=dt2, fontsize=15, color="C1", horizontalalignment="right")

		if param[k] in ["m", "M"] :
			ax[1][k+1].set_title("Difference between\n%ss" % (plotLabel[k]))
		else :
			ax[1][k+1].set_title("Difference between\n%s" % (plotLabel[k]))
		ax[1][k+1].set_xlabel("Difference")

	ax[0][1].set_ylabel("Density")
	ax[1][1].set_ylabel("Density")

	### Empty subplot
	ax[1][0].grid(False)
	ax[1][0].axis("off")


	plt.suptitle(graphInfo["title"], fontsize=20)

	plt.tight_layout(rect=[0, 0.03, 1, 0.9])

	plt.savefig("tmp/plot_%s_3.pdf" % (ID), orientation="landscape", format="pdf")
	plt.savefig("tmp/plot_%s_3.png" % (ID), orientation="landscape", format="png")

def pairwiseComparison(ID, x, y, x_infer, graphInfo, stanResult, priorInfo):
	plotLabel = [priorInfo["HDR"], priorInfo["LDR"], priorInfo["I"], priorInfo["S"]]

	pairwiseInference(ID, x[0], y[0], x_infer, graphInfo, stanResult, 1, priorInfo)
	pairwiseInference(ID, x[1], y[1], x_infer, graphInfo, stanResult, 2, priorInfo)

	fig, ax = plt.subplots(3, 3, figsize=(6, 6), sharex='col', sharey='row')

	for i in range(0, len(param)-2) :
		paramData1 = [stanResult["%s1" % (param[i])], stanResult["%s2" % (param[i])]]

		for j in range(i, len(param)-2) :
			paramData2 = [stanResult["%s1" % (param[j+1])], stanResult["%s2" % (param[j+1])]]

			#ax[j]
			ax[j][i].plot(paramData1[0], paramData2[0], 'o', color='C0', alpha=0.4)
			ax[j][i].plot(paramData1[1], paramData2[1], 'o', color='C1', alpha=0.4)
			ax[j][0].set_ylabel(plotLabel[j+1])

		ax[2][i].set_xlabel(plotLabel[i])

	ax[0][1].axis('off')
	ax[0][2].axis('off')
	ax[1][2].axis('off')

	plt.suptitle(graphInfo["title"], fontsize=20)

	plt.tight_layout(rect=[0, 0.03, 1, 0.9])

	plt.savefig("tmp/pairwise_%s_3A.pdf" % (ID), orientation="landscape", format="pdf")
	plt.savefig("tmp/pairwise_%s_3A.png" % (ID), orientation="landscape", format="png")

	fig, ax = plt.subplots(3, 3, figsize=(6, 6), sharex='col', sharey='row')

	for i in range(0, len(param)-2) :
		paramData1 = [stanResult["%s1" % (param[i])], stanResult["%s2" % (param[i])]]

		for j in range(i, len(param)-2) :
			paramData2 = [stanResult["%s1" % (param[j+1])], stanResult["%s2" % (param[j+1])]]

			#ax[j]
			ax[j][i].plot(paramData1[1], paramData2[1], 'o', color='C1', alpha=0.4)
			ax[j][i].plot(paramData1[0], paramData2[0], 'o', color='C0', alpha=0.4)
			ax[j][0].set_ylabel(plotLabel[j+1])

		ax[2][i].set_xlabel(plotLabel[i])

	ax[0][1].axis('off')
	ax[0][2].axis('off')
	ax[1][2].axis('off')

	plt.suptitle(graphInfo["title"], fontsize=20)

	plt.tight_layout(rect=[0, 0.03, 1, 0.9])

	plt.savefig("tmp/pairwise_%s_3B.pdf" % (ID), orientation="landscape", format="pdf")
	plt.savefig("tmp/pairwise_%s_3B.png" % (ID), orientation="landscape", format="png")
