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


cabin_data = total_data["Cabin"].str.extract('(?P<Cabin_Cap>^.)', expand=True)
total_data = pd.concat([total_data, cabin_data], axis=1)
total_data.head()


# In[11]:


total_data_cabin = total_data.groupby("Cabin_Cap")["Survived"].value_counts(dropna=False, normalize=True).unstack()
total_data_cabin


# In[12]:


total_data_cabin.fillna(0, inplace=True)
total_data_cabin.columns = ["nan", "d", "s"]
total_data_cabin["count"] = total_data_cabin.sum(axis=1)
total_data_cabin.sort_values("count", ascending=False)[["nan", "d", "s"]].plot.bar(stacked=True)
plt.savefig("cabin_cap.svg")
# B, D, E は生存率が高そうだが, 決定的な何かはない


# In[ ]:




