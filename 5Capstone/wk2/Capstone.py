# -*- coding: utf-8 -*-
"""
Created on Thurs Feb 23 20:55:11 2015

@author: EAmankwah
"""
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# any additional libraries would be imported here

ad = pd.read_csv('capstone.csv', low_memory=False)

#bug fix for display formats to avoid run time errors - put after code for loading data above
pd.set_option('display.float_format', lambda x:'%f'%x)

#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)

#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)

#setting variables you will be working with to numeric

#setting variables to numeric
for c in range(1,len(ad.columns)):
    ad[ad.columns[c]] = pd.to_numeric(ad[ad.columns[c]], errors='coerce')

print('Checking the column types:\n')
print('\n')
print(ad.dtypes)

#create a new variable to slit Income per person into lowI, mediumI and highI
ad['Ilevel'] = pd.cut(ad['x11_2013'],[200,3011,10505,71106],labels=['lowI', 'mediumI', 'highI'])

# Frequency distribution for the variable 'Ilevel'
print('\n')
print ('counts for Ilevel')
c_Ilevel = ad['Ilevel'].value_counts(sort=False)
print('\n')
print (c_Ilevel)

print('\n')
print('\n')
print ('percentages for Ilevel')

p_Ilevel = ad['Ilevel'].value_counts(sort=False,
normalize=True)
print('\n')
print (p_Ilevel)

print('\n')


# standard deviation and other descriptive statistics for quantitative variables
print ('describe income')
desc1 = ad['Ilevel'].describe()
print (desc1)

print ('median')
median1 = ad['x12_2013'].median()
print (median1)

sb.countplot(x='Ilevel',data=ad)
plt.xlabel('Income Levels')
plt.ylabel('Number of countries')
plt.title('Distribution of Income Levels by Countries')
plt.show()

'''#basic scatterplot:  Q->Q
scat1 = sb.regplot(x="incomeperperson", y="oilperperson", data=ad)
plt.xlabel('Income per Person')
plt.ylabel('Oil Consumption per Capita')
plt.title('Scatterplot for the Association Between Income per Person and Oil consumption per Capita')
plt.show()

scat2 = sb.regplot(x="incomeperperson", y="co2emissions", data=ad)
plt.xlabel('Income per Person')
plt.ylabel('Co2 Emissions')
plt.title('Scatterplot for the Association Between Income per Person and Co2 Emissions')
plt.show()

scat3 = sb.regplot(x="oilperperson", y="co2emissions", data=ad)
plt.xlabel('Oil Consumption per Capita')
plt.ylabel('Co2 Emissions')
plt.title('Scatterplot for the Association Between Oil Consumption per Capita and Co2 Emissions')
plt.show()

scat4 = sb.regplot(x="incomeperperson", y="hivrate", data=ad)
plt.xlabel('Income per Person')
plt.ylabel('HIV Rate')
plt.title('Scatterplot for the Association Between Income per Person and HIV Rate')
plt.show()

# quartile split (use qcut function & ask for 4 groups - gives you quartile split)
print ('Income per person - 4 categories - quartiles')
ad['INCOMEGRP4']=pd.qcut(ad.incomeperperson, 4, labels=["1=25th%tile","2=50%tile","3=75%tile","4=100%tile"])
'''
print('\n')
c6 = ad['Ilevel'].value_counts(sort=False, dropna=True)
print (c6)
print('\n')
# bivariate bar graph C->Q
sb.factorplot(x='Ilevel', y='x12_2013', data=ad, kind="bar", ci=None, size=7)
plt.xlabel('income group')
plt.ylabel('mean Co2 damage rate')
