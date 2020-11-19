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


emb_freq = total_data["Embarked"].value_counts()
emb_freq


# In[5]:


emb_freq.plot(kind="bar")


# In[6]:


total_data_emb = total_data.groupby("Embarked")["Survived"].value_counts(dropna=False).unstack()
total_data_emb


# In[7]:


total_data_emb.fillna(0, inplace=True)
total_data_emb.columns = ["nan", "d", "s"]
total_data_emb["count"] = total_data_emb.sum(axis=1)
total_data_emb.sort_values("count", ascending=False)[["nan", "d", "s"]].plot.bar(stacked=True)
plt.savefig("embarked.svg")


# In[ ]:




