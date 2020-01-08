#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # K Nearest Neighbors Project 
# 
# Welcome to the KNN Project! This will be a simple project very similar to the lecture, except you'll be given another data set. Go ahead and just follow the directions below.
# ## Import Libraries
# **Import pandas,seaborn, and the usual libraries.**

# In[37]:


import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt


# ## Get the Data
# ** Read the 'KNN_Project_Data csv file into a dataframe **

# In[2]:


df = pd.read_csv('KNN_Project_Data')


# **Check the head of the dataframe.**

# In[3]:


df.head()


# # EDA
# 
# Since this data is artificial, we'll just do a large pairplot with seaborn.
# 
# **Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.**

# In[4]:


sb.pairplot(df,hue='TARGET CLASS')


# # Standardize the Variables
# 
# Time to standardize the variables.
# 
# ** Import StandardScaler from Scikit learn.**

# In[7]:


from sklearn.preprocessing import StandardScaler


# ** Create a StandardScaler() object called scaler.**

# In[13]:


scaler = StandardScaler()


# ** Fit scaler to the features.**

# In[14]:


scaler.fit(df.drop('TARGET CLASS', axis=1))


# **Use the .transform() method to transform the features to a scaled version.**

# In[15]:


scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))


# **Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# In[17]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# # Train Test Split
# 
# **Use train_test_split to split your data into a training set and a testing set.**

# In[22]:


X = df_feat
y = df['TARGET CLASS']


# In[20]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# # Using KNN
# 
# **Import KNeighborsClassifier from scikit learn.**

# In[24]:


from sklearn.neighbors import KNeighborsClassifier


# **Create a KNN model instance with n_neighbors=1**

# In[25]:


knn = KNeighborsClassifier(n_neighbors=1)


# **Fit this KNN model to the training data.**

# In[26]:


knn.fit(X_train, y_train)


# # Predictions and Evaluations
# Let's evaluate our KNN model!

# **Use the predict method to predict values using your KNN model and X_test.**

# In[28]:


predictions = knn.predict(X_test)


# ** Create a confusion matrix and classification report.**

# In[30]:


from sklearn.metrics import classification_report,confusion_matrix


# In[32]:


print(confusion_matrix(y_test, predictions))


# In[33]:


print(classification_report(y_test,predictions))


# # Choosing a K Value
# Let's go ahead and use the elbow method to pick a good K Value!
# 
# ** Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list. Refer to the lecture if you are confused on this step.**

# In[36]:


error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(y_test != pred_i))


# **Now create the following plot using the information from your for loop.**

# In[51]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='--',marker='o',mfc='pink',mec='red')
plt.title('Error Report for variable K')
plt.xlabel('K')
plt.ylabel('Error')


# ## Retrain with new K Value
# 
# **Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.**

# In[50]:


knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
pred_30 = knn.predict(X_test)
print('K=30')
print('\n')
print(confusion_matrix(y_test,pred_30))
print('\n')
print(classification_report(y_test,pred_30))


# # Great Job!
