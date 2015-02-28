#Andrews Curve 
#Written by Ethan Spangler


import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import radviz
from pandas.tools.plotting import table 
from pandas import read_csv
from pandas.tools.plotting import andrews_curves
import os

#filepath ="/Users/ethanspangler/Desktop/MATH583/tonguedata.csv"
os.chdir(os.getcwd())
os.getcwd()

dc = read_csv("e-tongue2.csv",
header=0,
usecols=['Sample','GPS','STS','UMS','SPS','SWS','BRS'])

plt.figure()
andrews_curves(dc, 'Sample')
plt.show()