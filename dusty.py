###### IMPORT STATEMENTS
from __future__ import division
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
from scipy import stats

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


def cvTree(dataset, dep, n_bins = 10):
    if (str(type(dataset)) == "<class 'pandas.core.frame.DataFrame'>"):
        # Creating x and y matrices
        try:
            n = np.shape(dataset)[0]
            binset = n/n_bins
            y = pd.DataFrame()
            y = y.append(dataset[dep])
            y = y.T
            x = dataset.drop(dep, 1)
            nx = np.shape(x)[0]
            ny = np.shape(y)[0]
            resort1 = np.array(range(n))
            resort2 = np.array(np.random.random_sample(n))
            resort = np.array([resort1, resort2])
            resort = resort.T
            resort = sorted(resort, key = lambda x: x[1])
            resort = np.array(resort)

        except:
            print 'Invalid variable names'
            return None

        try:
            for i in range(n):
                resort[i,1] = resort[i,0]//binset
            resort = sorted(resort, key = lambda x: x[0])

        except:
            print 'Error in setting bins'
            return None

        try:
            clf = tree.DecisionTreeClassifier()
            acc = np.empty([n_bins,1])
            for i in range(n_bins):
                print i
                # Collate test sets
                x_train = pd.DataFrame()
                y_train = pd.DataFrame()
                x_test = pd.DataFrame()
                y_test = pd.DataFrame()
                for count in range(n):
                    if (resort[count][1]==i):
                        x_test = x_test.append(x.irow(count))
                        y_test = y_test.append(y.irow(count))
                    else:
                        x_train = x_train.append(x.irow(count))
                        y_train = y_train.append(y.irow(count))
                # Run Classifier
                clf.fit(x_train, y_train)
                pred = clf.predict(x_test)
                acc[[i]] = accuracy_score(y_test, pred)


        except:
            print 'Error in creating test and train sets'
            return None

        print acc


    else:
        print 'Data is not recognized as a Pandas Dataframe'
        return None 