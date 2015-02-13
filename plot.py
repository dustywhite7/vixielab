# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 09:48:33 2015

@author: dusty
"""

##---(Thu Feb 12 09:43:40 2015)--

from pandas.tools.plotting import radviz
import matplotlib.pyplot as plt
import pandas as pd
plt.figure()
data = pd.read_csv('/Users/Dusty/Documents/Machine Learning/Vixie/etongueData.csv')
#data = data.drop('Sample', 1)
data = data[['Sample', 'SRS', 'GPS', 'STS', 'UMS', 'SPS', 'SWS', 'BRS']]
radviz(data, 'Sample')
plt.show()
