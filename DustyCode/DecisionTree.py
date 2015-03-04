from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


os.chdir(os.getcwd())
os.getcwd()


data = pd.read_csv("etongueData.csv")
y = data.Sample
x = data.drop('Sample', 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)

clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)

pred = clf.predict(x_test)

acc = accuracy_score(y_test, pred)
print acc

tree.export_graphviz(clf, out_file='firstTree.dot')