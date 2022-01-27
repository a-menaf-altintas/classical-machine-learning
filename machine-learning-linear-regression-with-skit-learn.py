#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author: Abdulmenaf Altintas

# Machine learning models: Linear regression model with skit-learn


# In[2]:


import numpy as np
import math as mt
import pandas as pd
import quandl

import warnings
warnings.filterwarnings('ignore')


# In[3]:


# We wil build a model that predicts TESLA stock prices
df = quandl.get("WIKI/TSLA")

print(df.head())


# In[4]:


# We will select Adj. Open  Adj. High  Adj. Low  Adj. Close Adj. Volume as our starting features

df = df[['Adj. Open',  'Adj. Close', 'Adj. Low', 'Adj. High', 'Adj. Volume']]
print(df.head())


# In[5]:


# We can can create new features that will help our model to be more effective

df['open_close_percent'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df['high_low_percent'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. High'] * 100.0

# We replace 'Adj. High' and 'Adj. Low' with 'high_low_percent'. We also remove 'Adj. Open'.

df_new = df[['open_close_percent', 'high_low_percent', 'Adj. Close', 'Adj. Volume']]

print(df_new.head())


# In[6]:


# Determine prediction as label and put it into dataframe as new column
df_new.fillna(value=-10**4, inplace=True)
prediction = int(mt.ceil(0.01 * len(df)))

# define the label
df_new['label'] = df_new["Adj. Close"].shift(-prediction)
df_new.dropna(inplace=True) # drop Nan values
print(df_new.head())


# In[7]:


# import skit-learn modules

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


# In[8]:


# Define features and labels
X = np.array(df_new[['open_close_percent', 'high_low_percent', 'Adj. Close', 'Adj. Volume']])
y = np.array(df_new['label'])
print(X[0:2,:])
print(y[0:2])


# In[9]:


# Scale features between -1 and +1  before processing
X = preprocessing.scale(X)

# Start training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[10]:


# Choose classifier and fit
classifier = LinearRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print(score)

