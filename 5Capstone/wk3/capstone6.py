# -*- coding: utf-8 -*-
"""
Created on Mon March 3 12:20:10 2017

@author: EAmankwah
"""

#from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV

#Load the dataset
data = pd.read_csv("worldbank.csv")

#upper-case all DataFrame column names
#data.columns = map(str.upper, data.columns)

#clean data
data_clean = data[['x12_2013','x18_2013','x161_2013','x222_2013']].dropna()

#print(data_clean.dtypes)
print(data_clean.describe())
print('\n')

#select predictor variables and target variable as separate data sets
predvar= data_clean[['x12_2013','x18_2013','x161_2013','x222_2013']]

target = data_clean.x12_2013

# split data for scatterplots
train,test=train_test_split(data_clean, test_size=.4, random_state=123)

# better variable names and labels for plots
train['Carbon Dioxide Damage']=train['x12_2013']
train['Energy Depletion']=train['x18_2013']
train['Industry Value Added']=train['x161_2013']
train['Population']=train['x222_2013']

#scatterplot matrix for quantitative variables
fig1 = sns.PairGrid(train, y_vars=["Carbon Dioxide Damage"],
                 x_vars=['Energy Depletion','Industry Value Added',\
                 'Population'], palette="GnBu_d")
fig1.map(plt.scatter, s=50, edgecolor="white")
plt.title('Figure 1. Association Between Quantitative Predictors and Carbon Dioxide Damage',
                    fontsize = 12, loc='left')
fig1.savefig('reportfig1.jpg')

'''# boxplots for association between binary predictors & response
box1 = sns.boxplot(x="Equipment Failure", y="Manufacturing Lead Time", data=train)
plt.title('Figure 2. Association Between Equipment Failure and Manufacturing Lead Time',
                  fontsize = 12, loc='right')
box1 = plt.gcf()
box1.savefig('reportfig2.jpg')

box2 = sns.boxplot(x="Trainees Working", y="Manufacturing Lead Time", data=train)
plt.title('Figure 3. Association Between Trainee Involvement in Production\n and Manufacturing Lead Time',
                  fontsize = 12, multialignment='center')
box2 = plt.gcf()
box2.savefig('reportfig3.jpg')
'''
# standardize predictors to have mean=0 and sd=1 for lasso regression
predictors=predvar.copy()
from sklearn import preprocessing
predictors['x18_2013']=preprocessing.scale(predictors['x18_2013'].astype('float64'))
predictors['x161_2013']=preprocessing.scale(predictors['x161_2013'].astype('float64'))
predictors['x222_2013']=preprocessing.scale(predictors['x222_2013'].astype('float64'))

# split data into train and test sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target,
                                                              test_size=.4, random_state=123)
# specify the lasso regression model
model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)

# print variable names and regression coefficients
print(':: Regression Coefficents::')
print('\n')
print(dict(zip(predictors.columns, model.coef_)))
print('\n')

# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print('\n')
print(train_error)
print('\n')
print ('test data MSE')
print('\n')
print(test_error)
print('\n')

# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print('\n')
print(rsquared_train)
print('\n')
print ('test data R-square')
print('\n')
print(rsquared_test)
