# Bayesian inference for Dose-Response Analysis (BiDRA)

BiDRA is a user-friendly web interface for dose-response curves analysis. 

The interface is available through this URL: https://bidra.bioinfo.iric.ca/

We describe our Bayesian approach in this paper: https://academic.oup.com/bioinformatics/article/35/14/i464/5529233

### Branches
The code for BiDRA as presented at ISMB 2019 can be found in the ``ismb_2019``branch. Current developments are added to the ``master``branch.

### Dependencies
* Python3
* PyStan (and Stan)
* Flask
* Matplotlib
* Pickle

### Launch a local instance of BiDRA</h3>
1. Clone this repo
2. Create a ``tmp``directory to store data and figures
3. Within the ``stan`` repo, run ``compileStan.py`` to generate the compiled inference models
4. Launch the interface locally by running ``run.py``
