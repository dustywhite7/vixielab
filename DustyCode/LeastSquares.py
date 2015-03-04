from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from dusty import regOLS

os.chdir(os.getcwd())
os.getcwd()


data = pd.read_csv("etongueData.csv")
y = data.Sample
x = data.drop('Sample', 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)


reg = regOLS(data, 'ALCOHOL', ('SRS', 'GPS', 'STS', 'UMS', 'SPS', 'SWS', 'BRS'), intercept = 1)

print 'Covariate'.ljust(20) + 'Coefficient'.ljust(20) + 'Standard Error'.ljust(20) + 'T-Value'.ljust(20) + 'Pr(>|t|)'.ljust(20)
for i in range(7):
    print str(reg['Covariates'][i]).ljust(20) + str(reg['Coef'][i]).ljust(20) + str(reg['StdErr'][i]).ljust(20) + str(reg['Tstat'][i]).ljust(20) + str(reg['Pval'][i]).ljust(20)
    