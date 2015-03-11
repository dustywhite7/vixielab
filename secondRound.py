###### IMPORT STATEMENTS
from __future__ import division
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
import warnings

##### IMPORT DATA AND BREAK INTO TRAINING AND TESTING DATA

data = pd.read_csv('/Users/Dusty/Documents/Machine Learning/vixielab/etongueData.csv')

ylab = ('SRS','GPS','STS','UMS','SWS','BRS')
xlab = ('SOLIDS','PH','TA','ALCOHOL','LPP','SPP','TANNINS','PHENOLICS','PROTEINS','MANNOPR')

y = pd.DataFrame()

for i in ylab:
  y = y.append(data[i])

x = pd.DataFrame()

for i in xlab:
  x = x.append(data[i])

x = x.T
y = y.T

impval = np.zeros((len(xlab),1))
rsq = 0


##### RUN DECISION TREE REGRESSOR 1000 TIMES AND REPORT AVERAGE FEATURE IMPORTANCE

for time in range(1000):

  xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = .2)

  clf = tree.DecisionTreeRegressor()

  clf.fit(xtrain, ytrain)

  pred = clf.score(xtest, ytest)

  for count in range(len(xlab)):
    impval[count] = impval[count] + clf.feature_importances_[count]
  rsq = rsq + clf.score(xtest,ytest)


# xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = .2)

# clf = tree.DecisionTreeRegressor()

# clf.fit(xtrain, ytrain)

# pred = clf.score(xtest, ytest)

# impval = clf.feature_importances_
# rsq = clf.score(xtest,ytest)

print '\n'
print 'Average Feature Importance (Gini Importance)'
print '-'*40
for i in range(len(xlab)):
  print(xlab[i]).ljust(20) + str(impval[i]/1000).ljust(20)

print '-'*40
print 'Mean R-squared'.ljust(20) + str(rsq/1000)
print '\n'


