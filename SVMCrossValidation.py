
#import relevant libraries and assign them handles

import pandas as pd
import os 

#navigate working directory to where this file is saved
os.chdir(os.getcwd())
os.getcwd()

#this code assumes that you deleted the first row in the excel file
#so there are only variable names in the first row
data = pd.read_csv("e-tongue2.csv")
#type(data[0]) #gives the type of the element of the array
y = data.Sample
#x = pd.DataFrame(data,columns= collist)
x = data.drop('Sample',axis=1)


#Perform Support Vector Machine with Cross Validation
#http://scikit-learn.org/stable/modules/cross_validation.html
from sklearn import svm
from sklearn import cross_validation 

clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clf, x, y,cv=3)
print ""
print "Accuracy for 3 cross-validation groups:"
print "{0:.4f}, {1:.4f}, {2:.4f}".format(*scores)
#print("Accuracy for 3 cross-validation groups:/n %0.4f" % scores[1])
#type(scores[2])
print ""

print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 1.96))





