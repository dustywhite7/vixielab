###### IMPORT STATEMENTS
from __future__ import division
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import svm
import warnings

def dataprep(filespot, deplabel, testsize = .2):
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


def regOLS(dataset, dep, xvars, intercept=1):
    if (str(type(dataset)) == "<class 'pandas.core.frame.DataFrame'>"):

        # Creating x and y matrices
        try:
            n = np.shape(dataset)[0]
            k = np.shape(xvars)[0]
        except:
            print 'Dataset not imported correctly'
            return None
        try:
            y = dataset[str(dep)]
            x = pd.DataFrame()
        except:
            print 'Matrix y not created'
            return None
        try:
            for item in xvars:
                x = x.append(dataset[item])
            x = x.T
        except:
            print 'Matrix x not created'
            return None
        try:
            if (intercept==1):
                x['Intercept'] = np.ones(n)
                k = k+1
        except:
            print 'Intercept failed'
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
        except:
            print 'Cannot compute SE inner product'
            return None
        try:
            sighat = SSE/(n-k)
            sd = np.dot(sighat, np.linalg.inv(xtx))
        except:
            print 'Cannot calculate sigma hat'
            return None
        try:
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
                prob[i] = 1-stats.t.cdf(abs(tstat[i]), n-k)
        except:
            print 'Cannot calculate significance level'
            return None

        return {"Covariates": list(x.columns.values), "Coef": bhat, "StdErr": stdErr, "Tstat": tstat, "Pval": prob}
    else:
        print 'Data is not recognized as a Pandas Dataframe'
        return None


def cvTree(datafile, dep, observations, trials = 10, omit = None):
        try:
            accuracy = np.zeros((trials, 1))
            clf = tree.DecisionTreeClassifier()
            for i in range(10):
                x_train, x_test, y_train, y_test = balance_sets(datafile, dep, observations, omit)

                clf.fit(x_train, y_train)
                pred = clf.predict(x_test)

                accuracy[i] = accuracy_score(y_test, pred)
            return np.mean(accuracy)
        except:
            print "Error in running Cross-validation"



def cvSvmLin(datafile, dep, observations, trials = 10, omit = None):
        try:
            accuracy = np.zeros((trials, 1))
            clf = svm.SVC(kernel = 'linear', C = 100)
            for i in range(10):
                x_train, x_test, y_train, y_test = balance_sets(datafile, dep, observations, omit)

                clf.fit(x_train, y_train)
                pred = clf.predict(x_test)

                accuracy[i] = accuracy_score(y_test, pred)
            return np.mean(accuracy)
        except:
            print "Error in running Cross-validation"



def cvSvmPoly(datafile, dep, observations, trials = 10, omit = None):
        try:
            accuracy = np.zeros((trials, 1))
            clf = svm.SVC(kernel = 'poly', C = 100)
            for i in range(10):
                x_train, x_test, y_train, y_test = balance_sets(datafile, dep, observations, omit)

                clf.fit(x_train, y_train)
                pred = clf.predict(x_test)

                accuracy[i] = accuracy_score(y_test, pred)
            return np.mean(accuracy)
        except:
            print "Error in running Cross-validation"


def cvSvmRbf(datafile, dep, observations, trials = 10, omit = None):
        try:
            accuracy = np.zeros((trials, 1))
            clf = svm.SVC(kernel = 'rbf', C = 100)
            for i in range(10):
                x_train, x_test, y_train, y_test = balance_sets(datafile, dep, observations, omit)

                clf.fit(x_train, y_train)
                pred = clf.predict(x_test)

                accuracy[i] = accuracy_score(y_test, pred)
            return np.mean(accuracy)
        except:
            print "Error in running Cross-validation"


def cvNB(datafile, dep, observations, trials = 10, omit = None):
        try:
            accuracy = np.zeros((trials, 1))
            clf = GaussianNB()
            for i in range(10):
                x_train, x_test, y_train, y_test = balance_sets(datafile, dep, observations, omit)

                clf.fit(x_train, y_train)
                pred = clf.predict(x_test)

                accuracy[i] = accuracy_score(y_test, pred)
            return np.mean(accuracy)
        except:
            print "Error in running Cross-validation"


def dustyCV(datafile, dep, observations, trials = 10, omit = None):

    warnings.filterwarnings("ignore")
    acc = np.zeros((5,1))

    acc[0] = cvTree(datafile, dep, observations, trials, omit)
    acc[1] = cvSvmLin(datafile, dep, observations, trials, omit)
    acc[2] = cvSvmPoly(datafile, dep, observations, trials, omit)
    acc[3] = cvSvmRbf(datafile, dep, observations, trials, omit)
    acc[4] = cvNB(datafile, dep, observations, trials, omit)

    print "\n"
    print "-"*40
    print "Classifier".ljust(20) + "Accuracy Score".ljust(20)
    print "Decision Tree".ljust(20) + str(acc[0]).ljust(20)
    print "Linear SVM".ljust(20) + str(acc[1]).ljust(20)
    print "Polynomial SVM".ljust(20) + str(acc[2]).ljust(20)
    print "RBF SVM".ljust(20) + str(acc[3]).ljust(20)
    print "Naive Bayes".ljust(20) + str(acc[4]).ljust(20)
    print "-"*40
    print "All classifiers iterated " + str(trials) + " times."
    print "-"*40


    return acc

##### CREATE DATA THAT ENSURES SOME OF EACH GROUP IS IN BOTH TRAINING AND TESTING SETS

def balance_sets(filespot, deplabel, num_samp, omit = None):
    try:
        data = pd.read_csv(filespot)
    except:
        try:
            data = pd.read_csv(filespot)
        except:
            print "Data not in CSV or Excel format."
            return None, None, None, None

    try:
        if omit is not None:
            for m in range(len(omit)):
                data = data.drop(omit[m], 1)
        y = pd.DataFrame(data[deplabel].ravel())
        x = data.drop(deplabel, 1)
        x = pd.DataFrame(x)
    except:
        print "Invalid Dependent Variable"
        return None, None, None, None


    try:
        size = int(np.shape(data)[0]/num_samp)
        test = np.zeros((size,1))
        for i in range(size):
            test[i] = np.ceil(num_samp*np.random.random())
        xtest = pd.DataFrame()
        xtrain = pd.DataFrame()
        ytest = pd.DataFrame()
        ytrain = pd.DataFrame()
        for i in range(size):
            for j in range(num_samp):
                if (test[i]==j):
                    xtest = xtest.append(x.ix[i*3+j], 0)
                    ytest = ytest.append(y.ix[i*3+j], 0)
                else:
                    xtrain = xtrain.append(x.ix[i*3+j], 0)
                    ytrain = ytrain.append(y.ix[i*3+j], 0)
    except:
        print "Could not create training and testing data."
        return None, None, None, None




    return xtrain, xtest, ytrain, ytest
