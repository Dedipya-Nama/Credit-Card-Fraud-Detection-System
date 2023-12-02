#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


credit_card_data = pd.read_csv('creditcard.csv') #loading data to a pandas dataframe


# In[4]:


credit_card_data.head() #prints first five columns of the data


# In[9]:


credit_card_data.tail()


# In[10]:


credit_card_data.info()


# In[11]:


credit_card_data.isnull().sum()  #checking missing values in the dataframe


# In[12]:


#checking the distribution of valid and fraulent transactions

credit_card_data['Class'].value_counts()


# In[13]:


##apparently this dataset is highly unbalanced

##so we should seperate the data for analysis

valid = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[14]:


print(valid.shape)
print(fraud.shape)


# In[15]:


#statistical measures of the data
valid.Amount.describe()


# In[16]:


fraud.Amount.describe()


# In[17]:


credit_card_data.groupby('Class').mean()


# In[18]:


# under-sampling
# build a sample dataset contsainig similar distribution of normal transaction and fraudulent transactions

# The number of fraudulent transactions is 492

valid_sample = valid.sample(n=492) 


# In[19]:


# concatenating two dataframes


# In[20]:


new_dataset = pd.concat([valid_sample, fraud], axis =0)


# In[21]:


new_dataset.head()


# In[23]:


new_dataset.tail()


# In[24]:


new_dataset['Class'].value_counts()


# In[26]:


new_dataset.groupby('Class').mean()


# In[27]:


# splitting the data into features and targets
X = new_dataset.drop(columns ='Class', axis =1)
Y = new_dataset['Class']


# In[28]:


print(X)


# In[29]:


print(Y)


# In[31]:


# split the data into trsting and training data

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size =0.2, stratify = Y, random_state =2)


# In[33]:


print(X.shape, X_train.shape, X_test.shape)


# In[34]:


# model training
# logistic regression

model = LogisticRegression()

# training the logistic regression model with training data
model.fit(X_train, Y_train)


# In[37]:


# Model evaluation based on accuracy score

# accuracy on training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[41]:


print('Accuracy on training data :',training_data_accuracy)


# In[42]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[43]:


print('Accuracy score on test data :' , test_data_accuracy)


# In[ ]:




