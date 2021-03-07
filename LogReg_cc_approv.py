#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the necessary libraries for the project
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[11]:


# Importing the data and inspecting
cc_approvs = pd.read_csv(r'C:\Users\user\Desktop\CreditCardApprovals.csv', header = None)
cc_approvs.head()


# In[14]:


# Printing info of the DataFrame and summary statistics
print(cc_approvs.info())
print('\n')
print(cc_approvs.describe())

cc_approvs.tail(20)


# In[15]:


# Replacing '?' values with NaN values
cc_approvs = cc_approvs.replace('?', np.nan)
cc_approvs.tail(17)


# In[16]:


# Handling missing values with mean imputation
cc_approvs.fillna(cc_approvs.mean(), inplace = True)
print(cc_approvs.isnull().sum())


# In[18]:


#Imputing remaining missing values with the most frequent value
for col in cc_approvs.columns:
    if cc_approvs[col].dtypes == 'object':
        cc_approvs = cc_approvs.fillna(cc_approvs[col].value_counts().index[0])
        
print(cc_approvs.isnull().sum())


# In[19]:


#Numeric transformation with LabelEncoder
lab_encod = LabelEncoder()

for col in cc_approvs.columns:
    if cc_approvs[col].dtypes == 'object':
        # Using LabelEncoder to do the numeric transformation
        cc_approvs[col] = lab_encod.fit_transform(cc_approvs[col])


# In[21]:


#Splitting dataset into train and test set

#Dropping unimportant attributes like DriversLicense and ZipCode 
cc_approvs = cc_approvs.drop([11, 13], axis = 1)
cc_approvs_val = cc_approvs.values

X,y = cc_approvs_val[:,0:13] , cc_approvs_val[:,13]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)


# In[23]:


# Initializing MinMaxScaler and using it to rescale X_train and X_test

mm_scaler = MinMaxScaler(feature_range=(0, 1))
rescale_X_train = mm_scaler.fit_transform(X_train)
rescale_X_test = mm_scaler.fit_transform(X_test)


# In[24]:


# Initializing the logistic regression model and fitting it to the the train set

lr = LogisticRegression()
lr.fit(rescale_X_train , y_train)


# In[27]:


# Making predictions with logistic regression model
y_pred = lr.predict(rescale_X_test)

#Printing the accuracy of the logistic regression model
print('Accuracy of the model : ', lr.score(rescale_X_test, y_test))

#Printing confusion matrix results
confusion_matrix(y_test, y_pred)


# In[30]:


# Making the model perform better with GridSearchCV

tol = [0.01,0.001, 0.0001]
max_iter = [100, 150, 200]

# Creating a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict(tol = tol, max_iter = max_iter)

grid_CV = GridSearchCV(estimator = lr, param_grid = param_grid, cv = 7)

# Using MinMaxScaler to rescale X a
rescale_X = mm_scaler.fit_transform(X)

grid_result = grid_CV.fit(rescale_X, y)

best_score, best_params = grid_result.best_score_, grid_result.best_params_

print("Best: %f using %s" % (best_score, best_params))

