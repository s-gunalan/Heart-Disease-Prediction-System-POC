# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 14:33:51 2017

@author: 605482
"""

import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

dataset = genfromtxt("D:/guna/ML/POC/heart.csv",delimiter=',',skip_header=1)
imputer = Imputer()
dataset = imputer.fit_transform(dataset)
X=dataset[:,0:12]
Y=dataset[:,13]
pca=PCA(n_components=2,whiten=True).fit(X)
X_new=pca.transform(X)
target_names=['Non Diseased','Diseased']
"""def plot_2D(data, target, target_names):
     colors = cycle('rgbcmykw')
     target_ids = range(len(target_names))
     plt.figure()
     for i, c, label in zip(target_ids, colors, target_names):
         plt.scatter(data[target == i, 0], data[target == i, 1],
                    c=c, label=label)
     plt.legend()
     plt.savefig('Reduced_PCA_Graph')
plot_2D(X_new,Y,target_names)"""

#Logistic Regression
X_tr,X_te,Y_tr,Y_te=train_test_split(X,Y,test_size=0.2)
print X_tr.shape,Y_tr.shape,X_te.shape,Y_te.shape
lr=LogisticRegression()
model_lr=lr.fit(X_tr,Y_tr)
op_lrt=lr.predict(X_te)
print "Logistic Regression with split::"
res_logtt=confusion_matrix(Y_te,op_lrt)
sc_lrt=accuracy_score(Y_te,op_lrt)
print "Confusion Matrix:\n",res_logtt,"\nAccuracy:",accuracy_score(Y_te,op_lrt) ,"\nPrecision:",precision_score(Y_te,op_lrt) ,"\nRecall:",recall_score(Y_te,op_lrt),"\nF1 Score:" ,f1_score(Y_te,op_lrt)

#Without split
model_log=lr.fit(X_new,Y)
op_log=lr.predict(X_new)
print "Logistic Regression Without split::"
res_log=confusion_matrix(Y,op_log)
sc_log=accuracy_score(Y,op_log)
print "Confusion Matrix:\n",res_log,"\nAccuracy:",accuracy_score(Y,op_log) ,"\nPrecision:",precision_score(Y,op_log) ,"\nRecall:",recall_score(Y,op_log),"\nF1 Score:" ,f1_score(Y,op_log)

#Support Vector Machine
model_svmtt=LinearSVC(C=0.1)
model_svmtt=model_svmtt.fit(X_tr,Y_tr)
op_svmtt=model_svmtt.predict(X_te)
print "SVM with split:"
res_svmtt=confusion_matrix(Y_te,op_svmtt)
sc_svmtt=accuracy_score(Y_te,op_svmtt)
print "Confusion Matrix:\n",res_svmtt,"\nAccuracy:",accuracy_score(Y_te,op_svmtt) ,"\nPrecision:",precision_score(Y_te,op_svmtt) ,"\nRecall:",recall_score(Y_te,op_svmtt),"\nF1 Score:" ,f1_score(Y_te,op_svmtt)

#Without split
model_svm=LinearSVC(C=0.1)
model_svm=model_svm.fit(X_new,Y)
op_svm=model_svm.predict(X_new)
sc_svm=accuracy_score(Y,op_svm)
res_svm=confusion_matrix(Y,op_svm)
print "SVM without split:"
print "Confusion Matrix:\n",res_svm,"\nAccuracy:",accuracy_score(Y,op_svm) ,"\nPrecision:",precision_score(Y,op_svm) ,"\nRecall:",recall_score(Y,op_svm),"\nF1 Score:" ,f1_score(Y,op_svm)

#Naive Bayes
model_nbtt=GaussianNB()
model_nbtt=model_nbtt.fit(X_tr,Y_tr)
op_nbtt=model_nbtt.predict(X_te)
res_nbtt=confusion_matrix(Y_te,op_nbtt)
sc_nbtt=accuracy_score(Y_te,op_nbtt)
print "Naive Bayes with split::"
print "Confusion Matrix:\n",res_nbtt,"\nAccuracy:",accuracy_score(Y_te,op_nbtt) ,"\nPrecision:",precision_score(Y_te,op_nbtt) ,"\nRecall:",recall_score(Y_te,op_nbtt),"\nF1 Score:" ,f1_score(Y_te,op_nbtt)

#without Split
model_nb=GaussianNB()
model_nb=model_nb.fit(X_new,Y)
op_nb=model_nb.predict(X_new)
res_nb=confusion_matrix(Y,op_nb)
sc_nb=accuracy_score(Y,op_nb)
print "Naive Bayes without split::"
print "Confusion Matrix:\n",res_nb,"\nAccuracy:",accuracy_score(Y,op_nb) ,"\nPrecision:",precision_score(Y,op_nb) ,"\nRecall:",recall_score(Y,op_nb),"\nF1 Score:" ,f1_score(Y,op_nb)


#Random forest
model_rf=RandomForestClassifier(n_jobs=10,random_state=0)
model_rf=model_rf.fit(X_new,Y)
op_rf=model_rf.predict(X_new)
res_rf=confusion_matrix(Y,op_rf)
sc_rf=accuracy_score(Y,op_rf)
print "Random forest without split::"
print "Confusion Matrix:\n",res_rf,"\nAccuracy:",accuracy_score(Y,op_rf) ,"\nPrecision:",precision_score(Y,op_rf) ,"\nRecall:",recall_score(Y,op_rf),"\nF1 Score:" ,f1_score(Y,op_rf)

#with split
model_rft=RandomForestClassifier(n_jobs=10,random_state=0)
model_rft=model_rft.fit(X_tr,Y_tr)
op_rft=model_rft.predict(X_te)
res_rft=confusion_matrix(Y_te,op_rft)
sc_rft=accuracy_score(Y_te,op_rft)
print "Random forest without split::"
print "Confusion Matrix:\n",res_rft,"\nAccuracy:",accuracy_score(Y_te,op_rft) ,"\nPrecision:",precision_score(Y_te,op_rft) ,"\nRecall:",recall_score(Y_te,op_rft),"\nF1 Score:" ,f1_score(Y_te,op_rft)


#comparison
model=[]
model.append(sc_nb)
model.append(sc_nbtt)
model.append(sc_svm)
model.append(sc_svmtt)
model.append(sc_log)
model.append(sc_lrt)
model.append(sc_rf)
model.append(sc_rft)

