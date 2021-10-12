#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 18:21:49 2021

@author: rajkumar
"""

#import modules
import pandas  # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
#% matplotlib inline

#Loading Dataset
data=pandas.read_csv('/Users/rajkumar/Desktop/data-science-assignment/hyd.csv')
data.head()
#tail()returns last five observations.
data.tail()
data.info()

##Data Insights
candidateName = data.groupby('candidateName')
candidateName.mean()
data.describe()

##Data Visualization
candidateName_count=data.groupby('candidateName').count()
plt.bar(candidateName_count.index.values, candidateName_count['designation'])
plt.xlabel('companyName')
plt.ylabel('salary')
plt.show()
data.candidateName.value_counts()

##number of experience
experienceMas=data.groupby('experienceMas').count()
plt.bar(experienceMas.index.values, experienceMas['designation'])
plt.xlabel('experiencemas')
plt.ylabel('salary')
plt.show()

###category
Category=data.groupby('Category').count()
plt.bar(Category.index.values, Category['designation'])
plt.xlabel('Category')
plt.ylabel('salary')
plt.show()

##Subplots using Seaborn
features=[ 'companyName','designation','experienceMas','salary']
fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data = data)
    plt.xticks(rotation=90)
    plt.title("candidateName")
    
fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data = data, hue='left')
    plt.xticks(rotation=90)
    plt.title("salary")

##Cluster Analysis:
#import module
from sklearn.cluster import KMeans
# Filter data
companyName =  data[['candidateName', 'designation']][data.companyName == 1]
# Create groups using K-means clustering.
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(companyName)
# Add new column "label" annd assign cluster labels.
companyName['label'] = kmeans.labels_
# Draw scatter plot
plt.scatter(companyName['candidateName'], companyName['designation'], c=companyName['label'],cmap='Accent')
plt.xlabel('candidateName')
plt.ylabel('designation')
plt.title('3 Clusters of companyName')
plt.show()

##Building a Prediction Model
# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
data['salary']=le.fit_transform(data['salary'])
data['experienceMas ']=le.fit_transform(data['experienceMas'])

##Split Train and Test Set
#Spliting data into Feature and
X=data[['companyName', 'designation',
       'emailAddress', 'experienceMas', 'locationCurrentMas',
       'qualificationMas', 'qualificationMas2 ', 'salary', 'tel_other', 'Category']]
y=data['CandidateName']
# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training and 30% test

##Model Building
#Import Gradient Boosting Classifier model
from sklearn.ensemble import GradientBoostingClassifier

#Create Gradient Boosting Classifier
gb = GradientBoostingClassifier()

#Train the model using the training sets
gb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gb.predict(X_test)

##Evaluating Model Performance
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall
print("Recall:",metrics.recall_score(y_test, y_pred))
Accuracy: 0.971555555556
Precision: 0.958252427184
Recall: 0.920708955224

