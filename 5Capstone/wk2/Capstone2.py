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

'''# Frequency distribution for the variable 'Ilevel'
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
'''
desc1 = ad['x11_2013'].describe()
print (desc1)

desc2 = ad['x12_2013'].describe()
print (desc2)

#basic scatterplot:  Q->Q
scat1 = sb.regplot(x="x11_2013", y="x12_2013", data=ad)
plt.xlabel('Income of Countries')
plt.ylabel('Co2 Damage')
plt.title('Scatterplot for the Association Between Income and C02 damage')
plt.show()

scat2 = sb.regplot(x="x140_2013", y="x12_2013", data=ad)
plt.xlabel('Annual GDP Growth')
plt.ylabel('Co2 Damage')
plt.title('Scatterplot for the Association Between GDP growth and C02 damage')
plt.show()

scat3 = sb.regplot(x="x161_2013", y="x12_2013", data=ad)
plt.xlabel('Industry Value Added')
plt.ylabel('Co2 Damage')
plt.title('Scatterplot for the Association Between Industry and C02 damage')
plt.show()

scat4 = sb.regplot(x="x220_2013", y="x12_2013", data=ad)
plt.xlabel('Population Growth')
plt.ylabel('Co2 Damage')
plt.title('Scatterplot for the Association Between population growth and C02 damage')
plt.show()

scat5 = sb.regplot(x="x284_2013", y="x12_2013", data=ad)
plt.xlabel('Urbanization')
plt.ylabel('Co2 Damage')
plt.title('Scatterplot for the Association Between Urbanization and C02 damage')
plt.show()


'''
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
