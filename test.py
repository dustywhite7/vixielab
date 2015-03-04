###### IMPORT STATEMENTS
from __future__ import division
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
from scipy import stats
from dusty import balance_sets
from dusty import dustyCV

data = pd.read_csv('/Users/Dusty/Documents/Machine Learning/vixielab/etongueData.csv')

dustyCV('/Users/Dusty/Documents/Machine Learning/vixielab/etongueData.csv', 'Sample', 3, omit = ['GPS', 'SRS'])

# data = balance_sets('/Users/Dusty/Documents/Machine Learning/vixielab/etongueData.csv', 'Sample', 3, ['GPS', 'SRS'])
