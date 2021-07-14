#!/usr/bin/env python
# coding: utf-8

# ## <span style="color:blue"> Submitted By - SUMUNNA PAUL</span>
# 
# ## <span style="color:purple"> DATA SCIENCE AND ANALYTICS INTERN</span>
# 
# ## <span style="color:red"> THE SPARKS FOUNDATION INTERNSHIP PROGRAM JULY'21</span>
# 
# # Task:6 Prediction Using Decision Tree Algorithm
# 
# In this case study, my task is to create the decision tree classifier and visualize it graphically.
# 
# The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.
# 
# ## Reading the data into python
# The data has one file "Iris.csv". This file contains 150 flower species data.
# 
# ## Data description
# The business meaning of each column in the data is as below
# 
# * <b>SepalLengthCm</b>: Length of Sepal in cm
# * <b>SepalWidthCm</b>: Width of Sepal in cm
# * <b>PetalLengthCm</b>: Length of Petal in cm
# * <b>PetalWidthCm</b>: Width of Petal in cm
# * <b>Species</b>: Species of the flower

# In[1]:


# Reading the dataset
import numpy as np
import pandas as pd
Iris=pd.read_csv(filepath_or_buffer="C:/Users/Pratik/Desktop/IVY ProSchool/Sparks Internship/Iris.csv",sep=',', encoding='latin-1')
print('Shape of data before removing duplicate data :',Iris.shape)
Iris=Iris.drop_duplicates()
print('Shape of data after removing duplicate data :',Iris.shape)
Iris.head(10)


# ## Basic Data Exploration
# 
# Initial assessment of the data should be done to identify which columns are Quantitative, Categorical or Qualitative.

# In[2]:


Iris.head()


# In[3]:


Iris.info()


# In[4]:


Iris.describe(include='all')


# In[5]:


Iris.nunique()


# Based on the basic exploration above, you can now create a simple report of the data
# 
# * <b>Species</b>: Selected. Categorical. This is the <b>Target Variable!</b> 
# * <b>SepalLengthCm</b>: Selected. Continuous
# * <b>SepalWidthCm</b>: Selected. Continuous
# * <b>PetalLengthCm</b>: Selected. Continuous
# * <b>PetalWidthCm</b>: Selected. Continuous
# 
# ## Removing useless columns from the data

# In[2]:


Iris= Iris.drop(['Id'],axis=1)


# In[3]:


Iris.shape


# ## Missing values treatment
# Missing values are treated for each column separately.
# If a column has more than 30% data missing, then missing value treatment cannot be done.

# In[8]:


Iris.isnull().sum()


# There is no value missing in this data.
# 
# # Feature Selection (Anova Test)
# 
# Now its time to finally choose the best columns(Features) which are correlated to the Target variable.
# This can be done directly by ANOVA.
# When the target variable is Categorical and the predictor variable is Continuous we analyze the strength of relation using Anova test.
# 
# whether the four continuous variables 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm' and 'PetalWidthCm' are correlated with the target variable 'Species'or not, We can confirm this by looking at the results of ANOVA test below

# In[4]:


def AnovaFunction(inpData,TargetVariable,ContinuousVariables):
    from scipy.stats import f_oneway
    selectedcols=[]
    for predictor in ContinuousVariables:
        categorygroupedlist=inpData.groupby(TargetVariable)[predictor].apply(list)
        AnovaResult=f_oneway(*categorygroupedlist)
        if AnovaResult[1]<0.05 :
            print(predictor,'is co-related with' ,TargetVariable,'|P Value :',AnovaResult[1])
            selectedcols.append(predictor)
        else:
            print(predictor,'is not co-related with' ,TargetVariable,'|P Value :',AnovaResult[1])
    return selectedcols


# In[5]:


AnovaFunction(inpData=Iris,TargetVariable='Species',
              ContinuousVariables=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])


# The results of ANOVA confirm that these four predictors are correlated with Sprecies.
# 
# # Selecting final predictors for Machine Learning
# 
# Based on the above tests, selecting the final columns for machine learning

# In[15]:


selectedcols=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
DataforML= Iris[selectedcols]
DataforML.head()


# # Machine Learning: Splitting the data into Training and Testing sample
# We dont use the full data for creating the model. Some data is randomly selected and kept aside for checking how good the model is. This is known as Testing Data and the remaining data is called Training data on which the model is built. Typically 70% of data is used as Training data and the rest 30% is used as Tesing data.

# In[16]:


DataforML['Species']= Iris['Species']
DataforML.head()


# In[19]:


TargetVariable='Species'
Predictors=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X= DataforML[Predictors].values
y= DataforML[TargetVariable].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=428)


# # Standardization/Normalization of data
# 
# We can choose not to run this step if we want to compare the resultant accuracy of this transformation with the accuracy of raw data. 

# In[20]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
PredictorScaler= MinMaxScaler()
PredictorScalerFit= PredictorScaler.fit(X)
X= PredictorScalerFit.transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=42)


# In[21]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Logistic Regression
# 
# This classification algorithm is used when the value of the target variable is categorical in nature.

# In[25]:


from sklearn.linear_model import LogisticRegression
clf= LogisticRegression(C=1, penalty='l2', solver='newton-cg')
LOG= clf.fit(X_train,y_train)
Prediction= LOG.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test,Prediction))
print(metrics.confusion_matrix(y_test,Prediction))
F1_Score= metrics.f1_score(y_test,Prediction,average='weighted')
print('Accuracy of the Model on Test Data :',round(F1_Score,2))
from sklearn.model_selection import cross_val_score
AccuracyScore=cross_val_score(LOG,X,y,cv=10,scoring='f1_weighted')
print('Accuracy Values for 10-Fold Cross Validation :',AccuracyScore)
print('Final Accuracy of the Model :',round(AccuracyScore.mean(),2))


# # Decision Trees
# 
# It builds classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed.

# In[35]:


from sklearn import tree
clf= tree.DecisionTreeClassifier(max_depth=3,criterion='gini')
DTree= clf.fit(X_train,y_train)
Prediction= DTree.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test,Prediction))
print(metrics.confusion_matrix(y_test,Prediction))
F1_Score= metrics.f1_score(y_test,Prediction,average='weighted')
print('Accuracy of the Model on Test Data :',round(F1_Score,2))
get_ipython().run_line_magic('matplotlib', 'inline')
feature_importances=pd.Series(DTree.feature_importances_,index=Predictors)
feature_importances.nlargest(10).plot(kind='barh')
from sklearn.model_selection import cross_val_score
AccuracyScore=cross_val_score(DTree,X,y,cv=10,scoring='f1_weighted')
print('Accuracy Values for 10-Fold Cross Validation :',AccuracyScore)
print('Final Accuracy of the Model :',round(AccuracyScore.mean(),2))


# ## Visualize the Decision Tree

# In[33]:


import os
os.environ["PATH"] += os.pathsep + 'C:/ProgramData/Anaconda3/Library/bin/graphviz'


# In[37]:


from IPython.display import Image
from sklearn import tree
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=Predictors, class_names=TargetVariable)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png(), width=500,height=500)


# We can now feed any new/test data to this classifer and it would be able to predict the right class accordingly.
