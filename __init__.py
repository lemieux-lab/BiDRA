import os
from flask import Flask
from create import *

#app decorator
app = Flask(__name__)
app.config.from_object('config')