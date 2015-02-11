
# coding: utf-8

# In[2]:

#import relevant libraries and assign them handles

import pandas as pd

#this code assumes that you deleted the first row in the excel file
#so there are only variable names in the first row
data = pd.read_csv("C:\Users\Greg\Google Drive\WSU_S15\WineProblem\e-tongueDATA_Vixie.csv")
#type(data[0]) #gives the type of the element of the array
y = data.Sample
#x = pd.DataFrame(data,columns= collist)
x = data.drop('Sample',axis=1)



# In[3]:

#Perform Support Vector Machine with Cross Validation
#http://scikit-learn.org/stable/modules/cross_validation.html
from sklearn import svm
from sklearn import cross_validation 

clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clf, x, y,cv=3)
scores


# In[4]:

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:



