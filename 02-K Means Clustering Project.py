#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # K Means Clustering Project 
# 
# For this project we will attempt to use KMeans Clustering to cluster Universities into to two groups, Private and Public.
# 
# ___
# It is **very important to note, we actually have the labels for this data set, but we will NOT use them for the KMeans clustering algorithm, since that is an unsupervised learning algorithm.** 
# 
# When using the Kmeans algorithm under normal circumstances, it is because you don't have labels. In this case we will use the labels to try to get an idea of how well the algorithm performed, but you won't usually do this for Kmeans, so the classification report and confusion matrix at the end of this project, don't truly make sense in a real world setting!.
# ___
# 
# ## The Data
# 
# We will use a data frame with 777 observations on the following 18 variables.
# * Private A factor with levels No and Yes indicating private or public university
# * Apps Number of applications received
# * Accept Number of applications accepted
# * Enroll Number of new students enrolled
# * Top10perc Pct. new students from top 10% of H.S. class
# * Top25perc Pct. new students from top 25% of H.S. class
# * F.Undergrad Number of fulltime undergraduates
# * P.Undergrad Number of parttime undergraduates
# * Outstate Out-of-state tuition
# * Room.Board Room and board costs
# * Books Estimated book costs
# * Personal Estimated personal spending
# * PhD Pct. of faculty with Ph.D.’s
# * Terminal Pct. of faculty with terminal degree
# * S.F.Ratio Student/faculty ratio
# * perc.alumni Pct. alumni who donate
# * Expend Instructional expenditure per student
# * Grad.Rate Graduation rate

# ## Import Libraries
# 
# ** Import the libraries you usually use for data analysis.**

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# ## Get the Data

# ** Read in the College_Data file using read_csv. Figure out how to set the first column as the index.**

# In[6]:


college_data = pd.read_csv('College_Data')


# **Check the head of the data**

# In[7]:


college_data.head()


# ** Check the info() and describe() methods on the data.**

# In[8]:


college_data.info()


# In[9]:


college_data.describe()


# ## EDA
# 
# It's time to create some data visualizations!
# 
# ** Create a scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column. **

# In[14]:


sb.lmplot(x='Room.Board',y='Grad.Rate',data=college_data,hue='Private',fit_reg=False)


# **Create a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.**

# In[15]:


sb.lmplot(x='Outstate',y='F.Undergrad',data=college_data,hue='Private',fit_reg=False)


# ** Create a stacked histogram showing Out of State Tuition based on the Private column. Try doing this using [sns.FacetGrid](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.FacetGrid.html). If that is too tricky, see if you can do it just by using two instances of pandas.plot(kind='hist'). **

# In[32]:


g = sb.FacetGrid(college_data,hue="Private",palette='viridis',height=6,aspect=2)
g.map(plt.hist,'Outstate',bins=20,alpha=.7)


# **Create a similar histogram for the Grad.Rate column.**

# In[42]:


g = sb.FacetGrid(college_data,hue="Private",palette='viridis',height=6,aspect=2)
g.map(plt.hist,'Grad.Rate',bins=20,alpha=.7)


# ** Notice how there seems to be a private school with a graduation rate of higher than 100%.What is the name of that school?**

# In[35]:


college_data[college_data['Grad.Rate'] > 100]


# ** Set that school's graduation rate to 100 so it makes sense. You may get a warning not an error) when doing this operation, so use dataframe operations or just re-do the histogram visualization to make sure it actually went through.**

# In[43]:


college_data['Grad.Rate'][95] = 100


# In[44]:


g = sb.FacetGrid(college_data,hue="Private",palette='viridis',height=6,aspect=2)
g.map(plt.hist,'Grad.Rate',bins=20,alpha=.7)


# ## K Means Cluster Creation
# 
# Now it is time to create the Cluster labels!
# 
# ** Import KMeans from SciKit Learn.**

# In[50]:


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# ** Create an instance of a K Means model with 2 clusters.**

# In[46]:


km = KMeans(n_clusters=2)


# **Fit the model to all the data except for the Private label.**

# In[55]:


X = college_data.drop('Private',axis=1).drop('Unnamed: 0',axis=1)
km.fit(X)


# ** What are the cluster center vectors?**

# In[56]:


km.cluster_centers_


# ## Evaluation
# 
# There is no perfect way to evaluate clustering if you don't have the labels, however since this is just an exercise, we do have the labels, so we take advantage of this to evaluate our clusters, keep in mind, you usually won't have this luxury in the real world.
# 
# ** Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.**

# In[57]:


def convert(x):
    if x == 'Yes':
        return 1
    if x == 'No':
        return 0


# In[61]:


college_data['Private_binary']= college_data['Private'].apply(convert)


# In[62]:


college_data


# ** Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.**

# In[63]:


from sklearn.metrics import confusion_matrix, classification_report


# In[65]:


print(confusion_matrix(college_data['Private_binary'],km.labels_))


# In[66]:


print(classification_report(college_data['Private_binary'],km.labels_))


# Not so bad considering the algorithm is purely using the features to cluster the universities into 2 distinct groups! Hopefully you can begin to see how K Means is useful for clustering un-labeled data!
# 
# ## Great Job!
