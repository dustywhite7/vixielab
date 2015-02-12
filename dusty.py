###### IMPORT STATEMENTS
from __future__ import division
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
from scipy import stats

def dataprep(filespot, deplabel, testsize):
    try:
        data = pd.read_csv(filespot)
    except:
        try:
            data = pd.read_csv(filespot)
        except:
            print "Data not in CSV or Excel format."
            return None, None, None, None

    try:
        y = data[deplabel]
        x = data.drop(deplabel, 1)
    except:
        print "Invalid Dependent Variable"
        return None, None, None, None

    if ((testsize <=1) & (testsize >=0)):
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = testsize)
        return x_train, x_test, y_train, y_test
    else:
        print "Invalid Proportion for Test Set"

def runGaussian(x_train, x_test, y_train, y_test):
    clf = GaussianNB()
    try:
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)

        acc = accuracy_score(y_test, pred)

        print "Accuracy Score:" + str(acc)

    except:
        print "Invalid Data Provided for Naive Bayes"
        return 0
    return clf


def reg_OLS(dataset, dep, xvars, intercept):
    if (str(type(dataset)) == "<class 'pandas.core.frame.DataFrame'>"):

        # Creating x and y matrices
        try:
            n = np.shape(data)[0]
            k = np.size(xvars)
            y = dataset[dep]
            x = pd.DataFrame()
            for item in xvars:
                x = x.append(dataset[item])
            x = x.T
            if (intercept==1):
                x['Intercept'] = np.ones(n)
                k = k+1
        except:
            print 'Invalid variable names'
            return None

        # Generating Beta coefficients
        try:
            xtx = np.dot(x.T, x)
        except:
            print 'Cannot calculate inner product of x'
            return None
        try:
            bhat = np.dot(np.linalg.inv(xtx), np.dot(x.T, y))
        except:
            print 'Cannot calculate OLS coefficients'

        # Generating Standard Errors
        try:
            error = y - np.dot(x, bhat)
            SSE = np.dot(error.T, error)
            sighat = SSE/(n-k)
            sd = np.dot(sighat, np.linalg.inv(xtx))
            stdErr = np.zeros(shape=(k,1))
            for i in range(k):
                stdErr[i] = np.sqrt(sd[i][i])
        except:
            print 'Cannot calculate coefficient standard errors'
            return None

        # Generating T-statistics
        try:
            tstat = np.zeros(shape=(k,1))
            for i in range(k):
                tstat[i] = bhat[i]/stdErr[i]
        except:
            print 'Cannot calculate T-statistics'
            return None

        # Calculate Significance of Covariates
        try:
            prob = np.zeros(shape=(k,1))
            for i in range(k):
                prob[i] = 1-stats.t.cdf(abs(tstat[i]), 181)
        except:
            print 'Cannot calculate significance level'
            return None

        return {"Covariates": list(x.columns.values), "Coef": bhat, "StdErr": stdErr, "Tstat": tstat, "Pval": prob}
    else:
        print 'Data is not recognized as a Pandas Dataframe'
        return None
