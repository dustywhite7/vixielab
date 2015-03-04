from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

os.chdir(os.getcwd())
os.getcwd()


data = pd.read_csv("etongueData.csv")
y = data.Sample
x = data.drop('Sample', 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)

y2 = data.ALCOHOL
x2 = data.drop('Sample', 1)
x2 = x2.drop('ALCOHOL', 1)

x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size = .2)

clf = KMeans(n_clusters = 5)
clf.fit(x_train2, y_train2)

pred = clf.predict(x_test2)

print pred

plt.scatter(pred, x_test2[:,1])
plt.show()
