###### IMPORT STATEMENTS
from __future__ import division
from sklearn import tree
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
from sklearn import svm
from dusty import balance_sets

##### IMPORT DATA AND BREAK INTO TRAINING AND TESTING DATA

os.chdir(os.getcwd())
os.getcwd()



data = pd.read_csv("etongueData.csv")

data = data.drop(data.index[[22,47,133,170]])


ylab = ('SRS','GPS','STS','UMS','SPS','SWS','BRS')
xlab = ('SOLIDS','PH','TA','ALCOHOL','LPP','SPP','TANNINS','PHENOLICS','PROTEINS','MANNOPR')

y = data[['SRS','GPS','STS','UMS','SPS','SWS','BRS']]
x = data[['SOLIDS','PH','TA','ALCOHOL','LPP','SPP','TANNINS','PHENOLICS','PROTEINS','MANNOPR']]


##### CHOOSE HOW MANY TIMES TO RUN REGRESSOR
iter = 1


scores = np.zeros(1)
for i in range(1):
    xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = .1)







##### RANDOM FOREST REGRESSOR - BEST PERFORMANCE TO DATE - RSQ = .67

    clf = ensemble.RandomForestRegressor(n_estimators = 100, n_jobs = -1)
    clf.fit(xtrain, ytrain)

    scores[i] = clf.score(xtest,ytest)

print "The average R-squared from the Random Forest Regressors is: " + str(np.mean(scores)) + "\n"


print "The R-squared values from each Random Forest Regressor are: " + str(scores) + "\n"



##### THE SPECIAL ACCURACY SCORE DEPENDENT ON
##### THE ABILITY OF INDIVIDUALS TO DETECT THE
##### DIFFERENCE IN TASTE

pred = clf.predict(xtest)
acc = np.zeros(np.shape(ytest))


#  ENTER EPSILON VALUE (unit change that is the
#  most that a taste value could change without
#  people noticing the difference in flavor):

epsilon = 200


for i in range(np.shape(ytest)[0]):
    for j in range(np.shape(ytest)[1]):
        if abs(pred[i][j]-ytest[i][j]) < epsilon:
            acc[i][j]=1

accrow = np.zeros(np.shape(ytest)[0])

for i in range(np.shape(ytest)[0]):
    accrow[i] = np.mean(acc[i,:])


acccol = np.zeros(np.shape(ytest)[1])

for i in range(np.shape(ytest)[1]):
    acccol[i] = np.mean(acc[:,i])

acc = np.mean(accrow)

print "The accuracy of each row based on provided epsilon value is: "
print "-"*30
for i in range(np.shape(ytest)[0]):
  print "Row " + str(i) + ": ".ljust(5) + str(accrow[i])
print "-"*30

print "The accuracy of each taste parameter based on provided epsilon value is: "
print "-"*30
for i in range(np.shape(ytest)[1]):
  print str(ylab[i]) + ": ".ljust(5) + str(acccol[i])
print "-"*30

