#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  10 07:02:12 2018

@author: hurenjie
"""

from sklearn import datasets
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

#Part 1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
train_scores=[]
test_scores=[]
a=range(1,11)
for i in a:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, 
                                                        random_state=i, 
                                                        stratify=y)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    
    tree = DecisionTreeClassifier(criterion='gini',
                                      max_depth=4,
                                      random_state=1)
    tree.fit(X_train, y_train)
    y_train_pred=tree.predict(X_train)
    y_test_pred=tree.predict(X_test)
    train_scores.append(metrics.accuracy_score(y_train,y_train_pred))
    test_scores.append(metrics.accuracy_score(y_test,y_test_pred))
print("train accuracy score:\n{}".format(train_scores))
print("\n")
print("mean of train accuracy score:\n{} ".format(np.mean(train_scores)))
print("\n")
print("std of train accuracy score:\n{}".format(np.std(train_scores)))
print("\n")

print("test accuracy score:\n{}".format(test_scores))
print("\n")
print("mean of test accuracy score:\n{} ".format(np.mean(test_scores)))
print("\n")
print("std of test accuracy score:\n{}".format(np.std(test_scores)))
print("\n")

#Part 2
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=tree,
                             X=X_train,
                             y=y_train,
                             cv=10,
                            n_jobs=1)
print('CV accuracy scores: %s' % scores)
print("\n")
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
          np.std(scores)))
print("My name is RENJIE HU")
print("My NetID is: 659740767")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")