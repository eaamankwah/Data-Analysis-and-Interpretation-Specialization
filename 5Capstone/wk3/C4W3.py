# -*- coding: utf-8 -*-
"""
Created on Sun Dec 6 23:00:54 2016

@author: EAmankwah
"""

# -*- coding: utf-8 -*-

#from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV

# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%.2f'%x)

#os.chdir("C:\Decission TREES")

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

#upper-case all DataFrame column names
ad.columns = map(str.upper, ad.columns)

#clean data
data_clean = ad[['INCOMEPERPERSON','POLITYSCORE','CO2EMISSIONS','OILPERPERSON','HIVRATE','ALCCONSUMPTION',
'ARMEDFORCESRATE','FEMALEEMPLOYRATE','RELECTRICPERPERSON',
'URBANRATE','EMPLOYRATE','INTERNETUSERATE','LIFEEXPECTANCY']].dropna()

#print(data_clean.dtypes)
#print(data_clean.describe())

# categories one predictor variable, polityscore into binary variable
def INTERNETGRP (row):
   if row['INTERNETUSERATE'] <=46.68:
      return 0
   if row['INTERNETUSERATE']>46.68:
      return 1

#checking that recoding is dane
data_clean['INTERNETGRP'] = data_clean.apply(lambda row:INTERNETGRP(row), axis=1)
chkld = data_clean['INTERNETGRP'].value_counts(sort=False, dropna=False)
print(chkld)

"""
Modeling and Prediction
"""
#select predictor variables and target variable as separate data sets

predvar = data_clean[['POLITYSCORE','CO2EMISSIONS','OILPERPERSON','HIVRATE','ALCCONSUMPTION',
'ARMEDFORCESRATE','FEMALEEMPLOYRATE','RELECTRICPERPERSON',
'URBANRATE','EMPLOYRATE','INTERNETGRP','LIFEEXPECTANCY']]

target = data_clean.INCOMEPERPERSON
# standardize predictors to have mean=0 and sd=1
predictors=predvar.copy().dropna()
from sklearn import preprocessing
#predictors['OILPERPERSON']=preprocessing.scale(predictors['OILPERPERSON'].astype('float64'))
#predictors['HIVRATE']=preprocessing.scale(predictors['HIVRATE'].astype('float64'))
#predictors['ALCCONSUMPTION']=preprocessing.scale(predictors['ALCCONSUMPTION'].astype('float64'))
#predictors['ARMEDFORCESRATE']=preprocessing.scale(predictors['ARMEDFORCESRATE'].astype('float64'))
#predictors['BREASTCANCERPER100th']=preprocessing.scale(predictors['BREASTCANCERPER100th'].astype('float64'))
#predictors['FEMALEEMPLOYRATE']=preprocessing.scale(predictors['FEMALEEMPLOYRATE'].astype('float64'))
#predictors['RELECTRICPERPERSON']=preprocessing.scale(predictors['RELECTRICPERPERSON'].astype('float64'))
#predictors['URBANRATE']=preprocessing.scale(predictors['URBANRATE'].astype('float64'))
#predictors['POLITYSCORE']=preprocessing.scale(predictors['POLITYSCORE'].astype('float64'))
#predictors['SUICIDEPER100th']=preprocessing.scale(predictors['SUICIDEPER100th'].astype('float64'))
#predictors['EMPLOYRATE']=preprocessing.scale(predictors['EMPLOYRATE'].astype('float64'))
#predictors['INTERNETGRP']=preprocessing.scale(predictors['INTERNETGRP'].astype('float64'))
#predictors['LIFEEXPECTANCY']=preprocessing.scale(predictors['LIFEEXPECTANCY'].astype('float64'))
#predictors['CO2EMISSIONS']=preprocessing.scale(predictors['CO2EMISSIONS'].astype('float64'))

#looping through
for col in ['OILPERPERSON','HIVRATE','ALCCONSUMPTION','ARMEDFORCESRATE','FEMALEEMPLOYRATE','RELECTRICPERPERSON',
'URBANRATE','POLITYSCORE','EMPLOYRATE','INTERNETGRP','LIFEEXPECTANCY','CO2EMISSIONS']:
    preprocessing.scale(predictors[col].astype('float64'))

# split data into train and test sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target,
                                                              test_size=.3, random_state=123)

# specify the lasso regression model
model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)

# print variable names and regression coefficients
print(dict(zip(predictors.columns, model.coef_)))

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
print(train_error)
print ('test data MSE')
print(test_error)

# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)
