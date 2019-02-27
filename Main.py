#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 17:04:23 2019

@author: srinivas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


dataset=pd.read_csv('bank-additional-full.csv',sep=';')


inputs=np.array(dataset)

X=inputs[:,0:20]
Y=inputs[:,20]
Y=np.reshape(Y,(-1,1))

labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X1.fit_transform(X[:, 2])
X[:, 3] = labelencoder_X1.fit_transform(X[:, 3])
X[:, 4] = labelencoder_X1.fit_transform(X[:, 4])
X[:, 5] = labelencoder_X1.fit_transform(X[:, 5])
X[:, 6] = labelencoder_X1.fit_transform(X[:, 6])
X[:, 7] = labelencoder_X1.fit_transform(X[:, 7])
X[:, 8] = labelencoder_X1.fit_transform(X[:, 8])
X[:, 9] = labelencoder_X1.fit_transform(X[:, 9])
X[:, 14] = labelencoder_X1.fit_transform(X[:, 14])

labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)
Y=np.reshape(Y,(-1,1))


onehotencoder = OneHotEncoder(categorical_features = [1,2,3,4,5,6,7,8,9,14])
X = onehotencoder.fit_transform(X).toarray()

Scx=StandardScaler()
X[:,53:55]=Scx.fit_transform(X[:,53:55])
X[:,56:57]=Scx.fit_transform(X[:,56:57])
X[:,59:61]=Scx.fit_transform(X[:,59:61])
X[:,62:63]=Scx.fit_transform(X[:,62:63])


dataset_output=pd.read_csv('bank-additional.csv',sep=';')


outputs=np.array(dataset_output)

X_out=outputs[:,0:20]
Y_out=outputs[:,20]
Y_out=np.reshape(Y_out,(-1,1))

labelencoder_X1 = LabelEncoder()
X_out[:, 1] = labelencoder_X1.fit_transform(X_out[:, 1])
X_out[:, 2] = labelencoder_X1.fit_transform(X_out[:, 2])
X_out[:, 3] = labelencoder_X1.fit_transform(X_out[:, 3])
X_out[:, 4] = labelencoder_X1.fit_transform(X_out[:, 4])
X_out[:, 5] = labelencoder_X1.fit_transform(X_out[:, 5])
X_out[:, 6] = labelencoder_X1.fit_transform(X_out[:, 6])
X_out[:, 7] = labelencoder_X1.fit_transform(X_out[:, 7])
X_out[:, 8] = labelencoder_X1.fit_transform(X_out[:, 8])
X_out[:, 9] = labelencoder_X1.fit_transform(X_out[:, 9])
X_out[:, 14] = labelencoder_X1.fit_transform(X_out[:, 14])

labelencoder_y = LabelEncoder()
Y_out = labelencoder_y.fit_transform(Y_out)
Y_out=np.reshape(Y_out,(-1,1))


onehotencoder = OneHotEncoder(categorical_features = [1,2,3,4,5,6,7,8,9,14])
X_out = onehotencoder.fit_transform(X_out).toarray()

Scx=StandardScaler()
X_out[:,53:55]=Scx.fit_transform(X_out[:,53:55])
X_out[:,56:57]=Scx.fit_transform(X_out[:,56:57])
X_out[:,59:61]=Scx.fit_transform(X_out[:,59:61])
X_out[:,62:63]=Scx.fit_transform(X_out[:,62:63])

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(X,Y)

y_pred_dt=classifier.predict(X_out)

confusion_matrix(Y_out,y_pred_dt)

f1_score(Y_out,y_pred_dt)




