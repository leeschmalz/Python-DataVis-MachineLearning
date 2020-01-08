#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Plotly and Cufflinks

# Plotly is a library that allows you to create interactive plots that you can use in dashboards or websites (you can save them as html files or static images).
# 
# ## Installation
# 
# In order for this all to work, you'll need to install plotly and cufflinks to call plots directly off of a pandas dataframe. These libraries are not currently available through **conda** but are available through **pip**. Install the libraries at your command line/terminal using:
# 
#     pip install plotly
#     pip install cufflinks
# 
# ** NOTE: Make sure you only have one installation of Python on your computer when you do this, otherwise the installation may not work. **
# 
# ## Imports and Set-up

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print(__version__) # requires version >= 1.9.0


# In[3]:


import cufflinks as cf


# In[4]:


# For Notebooks
init_notebook_mode(connected=True)


# In[5]:


# For offline use
cf.go_offline()


# ### Fake Data

# In[6]:


df = pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())


# In[22]:


df.head()


# In[39]:


df2 = pd.DataFrame({'Category':['A','B','C'],'Values':[32,43,50]})


# In[40]:


df2.head()


# ## Using Cufflinks and iplot()
# 
# * scatter
# * bar
# * box
# * spread
# * ratio
# * heatmap
# * surface
# * histogram
# * bubble

# ## Scatter

# In[20]:


df.iplot(kind='scatter',x='A',y='B',mode='markers',size=10)


# ## Bar Plots

# In[28]:


df2.iplot(kind='bar',x='Category',y='Values')


# In[33]:


df.count().iplot(kind='bar')


# ## Boxplots

# In[34]:


df.iplot(kind='box')


# ## 3d Surface

# In[47]:


df3 = pd.DataFrame({'x':[1,2,3,4,5],'y':[10,20,30,20,10],'z':[5,4,3,2,1]})
df3.iplot(kind='surface',colorscale='rdylbu')


# ## Spread

# In[50]:


df[['A','B']].iplot(kind='spread')


# ## histogram

# In[56]:


df['A'].iplot(kind='hist',bins=25)


# In[60]:


df.iplot(kind='bubble',x='A',y='B',size='C')


# ## scatter_matrix()
# 
# Similar to sns.pairplot()

# In[13]:


df.scatter_matrix()


# # Great Job!
