#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 23:09:19 2021

@author: jayanthvasanthkumar
"""

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

import pandas as pd
from pandas.plotting import parallel_coordinates

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#Reading the data from the csv, loading into a DataFrame object
dataset = pd.read_csv('diabetes_symptoms_dataset.csv')
headers = pd.read_csv('diabetes_symptoms_dataset.csv',header=None).iloc[0, :-1].values
#Segmenting the dataset into the patient features (Y) matrix and result (y) vector
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Training to testing split set to 75:25 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Instantiate classifier object as a Bernoulli Naive Bayes model
classifier = BernoulliNB()

#===== FITTING PROCESS =====
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

def predict_result(patient_symptoms):
    return "+VE" if classifier.predict([patient_symptoms]) else "-VE"

def obtain_symptoms_arr():
    symptoms_arr = []
    for value in headers:
        usr_inp = int(input("Enter the (value of) "+value+": "))
        symptoms_arr.append(usr_inp)
    return symptoms_arr

#TEST PATIENT FEATURE VECTOR; can be replaced for manual/custom input by admin
test_patient_symptoms_1 = [46,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0]
test_patient_symptoms_2 = [77,0,1,1,1,1,0,1,0,1,1,0,1,0,0,0]
#print("Preduction result patient 1: ",predict_result(test_patient_symptoms_1))
#print("Preduction result patient 2: ",predict_result(test_patient_symptoms_2))

symptoms_arr = obtain_symptoms_arr()
print("Preduction result USER INPUT: ",predict_result(symptoms_arr))

#Instantiate confusion matrix of the trained model
matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n",matrix)
print("\nAccuracy score: ",round(accuracy_score(y_test, y_pred),3)*100,"%")

#====== VISUALISAING =======
import seaborn as sns

#Utilising the lmplot method from the seaborn class to pass in the chosen parameters
#to generate the appropriate graph for binary features
#y-jitter of 0.11 chosen to spread out data points in the vertical axis such to improve
#the clarity of each of the class' population and density
sns.lmplot('Age', 'Polyuria', dataset, hue='class', fit_reg=False, y_jitter=(.11))
fig = plt.gcf()
fig.set_size_inches(10, 5)
plt.show()

#parallel_coordinates(dataset.iloc[:, 1:],'class',color=['red','green'])
#plt.show()
'''
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 2, stop = X_set[:, 1].max() + 2, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Gender')
plt.legend()
plt.show()
'''










