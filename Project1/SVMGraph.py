#SVM classification and graphing

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn import cross_validation 
import os

#navigate working directory to where this file is saved
os.chdir(os.getcwd())
os.getcwd()

# import some data to play with
df = pd.read_csv("e-tongue2.csv")

dfy = df.ix[:,'Sample']
#x = pd.DataFrame(data,columns= collist)
dfx = df.ix[:, 'SRS':'GPS'] # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
X = dfx.as_matrix()
y = dfy.as_matrix()

#print type(data)
#print type(df)
#print type(data2)
#print type(dfy)
#print type(dfx)
#print type(X)
#print type(y)

 

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)


# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# step size in the mesh
h = ((x_max - x_min)+(y_max - y_min))/250 

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
					 
# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel']


#for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
#for i, clf in enumerate((svc,rbf_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
#    plt.subplot(2, 2, i + 1)
#    plt.subplots_adjust(wspace=0.4, hspace=0.4)

 #   Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('SRS')
plt.ylabel('GPS')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title(titles[1])



plt.show()