#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Logistic Regression Project 
# 
# In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ## Import Libraries
# 
# **Import a few libraries you think you'll need (Or just import them as you go along!)**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# ## Get the Data
# **Read in the advertising.csv file and set it to a data frame called ad_data.**

# In[2]:


ad_data = pd.read_csv('advertising.csv')


# **Check the head of ad_data**

# In[3]:


ad_data.head()


# ** Use info and describe() on ad_data**

# In[4]:


ad_data.describe()


# In[5]:


ad_data.info()


# ## Exploratory Data Analysis
# 
# Let's use seaborn to explore the data!
# 
# Try recreating the plots shown below!
# 
# ** Create a histogram of the Age**

# In[6]:


sb.distplot(ad_data['Age'])


# **Create a jointplot showing Area Income versus Age.**

# In[7]:


sb.jointplot('Age','Area Income',ad_data)


# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**

# In[11]:


sb.jointplot('Age','Daily Time Spent on Site',ad_data,kind='kde',color='red')


# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

# In[13]:


sb.jointplot('Daily Time Spent on Site','Daily Internet Usage',ad_data,color='green')


# ** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

# In[16]:


sb.pairplot(ad_data,hue='Clicked on Ad')


# # Logistic Regression
# 
# Now it's time to do a train test split, and train our model!
# 
# You'll have the freedom here to choose columns that you want to train on!

# ** Split the data into training set and testing set using train_test_split**

# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X=ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y=ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[89]:





# ** Train and fit a logistic regression model on the training set.**

# In[19]:


from sklearn.linear_model import LogisticRegression


# In[21]:


logisticModel = LogisticRegression()


# In[22]:


logisticModel.fit(X_train,y_train)


# ## Predictions and Evaluations
# ** Now predict values for the testing data.**

# In[26]:


predict = logisticModel.predict(X_test)


# ** Create a classification report for the model.**

# In[30]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[28]:


print(classification_report(y_test,predict))


# In[38]:


print(np.matrix([['TN','FP'], ['FN','TP']]))
print(confusion_matrix(y_test,predict))


# ## Great Job!
