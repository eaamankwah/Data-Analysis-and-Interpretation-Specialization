# -*- coding: utf-8 -*-
"""
Created on Sun Dec 4 22:05:54 2016

@author: EAmankwah
"""

# -*- coding: utf-8 -*-

from pandas import Series, DataFrame
import pandas as pd
from sklearn.cross_validation import train_test_split
import sklearn.metrics
from sklearn import tree

# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%.2f'%x)

#os.chdir("C:\Decission TREES")

"""
Data Engineering and Analysis
"""
#Load the dataset

ad = pd.read_csv('research_data.csv')

#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)

#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)

# convert variables to numeric format using convert_objects function

ad['incomeperperson'] = pd.to_numeric(ad['incomeperperson'], errors='coerce')
ad['co2emissions'] = pd.to_numeric(ad['co2emissions'], errors='coerce')
ad['oilperperson'] = pd.to_numeric(ad['oilperperson'], errors='coerce')
ad['relectricperperson'] = pd.to_numeric(ad['relectricperperson'], errors='coerce')
ad['urbanrate'] = pd.to_numeric(ad['urbanrate'], errors='coerce')
ad['internetuserate'] = pd.to_numeric(ad['internetuserate'], errors='coerce')
ad['hivrate'] = pd.to_numeric(ad['hivrate'], errors='coerce')
ad['lifeexpectancy'] = pd.to_numeric(ad['lifeexpectancy'], errors='coerce')
ad['alcconsumption'] = pd.to_numeric(ad['alcconsumption'], errors='coerce')
ad['armedforcesrate'] = pd.to_numeric(ad['armedforcesrate'], errors='coerce')
ad['breastcancerper100th'] = pd.to_numeric(ad['breastcancerper100th'], errors='coerce')
ad['femaleemployrate'] = pd.to_numeric(ad['femaleemployrate'], errors='coerce')
ad['polityscore'] = pd.to_numeric(ad['polityscore'], errors='coerce')
ad['suicideper100th'] = pd.to_numeric(ad['suicideper100th'], errors='coerce')
ad['employrate'] = pd.to_numeric(ad['employrate'], errors='coerce')

print('\n')

data_clean = ad[['incomeperperson','co2emissions','oilperperson','hivrate','alcconsumption',
'armedforcesrate','breastcancerper100th','femaleemployrate','relectricperperson',
'urbanrate','polityscore','suicideper100th','employrate','internetuserate','lifeexpectancy']].dropna()

print(data_clean.dtypes)
print(data_clean.describe())

# categories response variable into binary variable
def internetgrp (row):
   if row['internetuserate'] <=46.68:
      return 0
   if row['internetuserate']>46.68:
      return 1

#checking that recoding is dane
data_clean['internetgrp'] = data_clean.apply(lambda row:internetgrp(row), axis=1)
chkld = data_clean['internetgrp'].value_counts(sort=False, dropna=False)
print(chkld)

"""
Modeling and Prediction
"""
#Split into training and testing sets

predictors = data_clean[['incomeperperson','co2emissions','hivrate',
'lifeexpectancy','oilperperson','relectricperperson','urbanrate']]

targets = data_clean.internetgrp

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=0.4)

print(pred_train.shape)
print(pred_test.shape)
print(tar_train.shape)
print(tar_test.shape)

#Build model on training data
clf=tree.DecisionTreeClassifier()
clf=clf.fit(pred_train,tar_train)

predictions=clf.predict(pred_test)

print(sklearn.metrics.confusion_matrix(tar_test,predictions))
print(sklearn.metrics.accuracy_score(tar_test, predictions))

#Displaying the decision tree
from io import StringIO
from IPython.display import Image
out = StringIO()
tree.export_graphviz(clf, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
img=Image(graph.create_png())

