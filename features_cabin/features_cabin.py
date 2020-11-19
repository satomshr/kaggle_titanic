#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


train_data = pd.read_csv("../train.csv")
test_data = pd.read_csv("../test.csv")
total_data = pd.concat([train_data, test_data])


# In[3]:


total_data.describe(include="O")


# In[4]:


cabin_freq = total_data["Cabin"].value_counts()
cabin_freq


# In[5]:


# https://yolo.love/pandas/series/#6
cabin_freq[cabin_freq > 3]


# In[6]:


cabin_freq[cabin_freq > 1].plot(figsize=(15,10), kind="bar")


# In[7]:


total_data_cabin = total_data.groupby("Cabin")["Survived"].value_counts(dropna=False).unstack()
total_data_cabin


# In[8]:


total_data_cabin.fillna(0, inplace=True)
total_data_cabin.columns = ["nan", "d", "s"]
total_data_cabin["count"] = total_data_cabin.sum(axis=1)
total_data_cabin[total_data_cabin["count"] > 2].sort_values("count", ascending=False)[["nan", "d", "s"]].plot.bar(figsize=(15,10),stacked=True)
plt.savefig("cabin.svg")


# In[14]:


total_data[total_data["Cabin"]=="C23 C25 C27"]


# In[15]:


total_data[total_data["Cabin"]=="B57 B59 B63 B66"]


# In[16]:


total_data[total_data["Cabin"]=="G6"]


# In[17]:


total_data[total_data["Cabin"]=="B96 B98"]


# In[18]:


total_data[total_data["Cabin"]=="C22 C26"]


# In[19]:


total_data[total_data["Cabin"]=="C78"]


# In[ ]:




