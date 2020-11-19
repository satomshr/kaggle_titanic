#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt


# In[2]:


train_data = pd.read_csv("../train.csv")
test_data = pd.read_csv("../test.csv")
total_data = pd.concat([train_data, test_data])


# In[3]:


total_data.describe(include="O")


# In[4]:


ticket_freq = total_data["Ticket"].value_counts()


# In[5]:


ticket_freq


# In[6]:


# https://yolo.love/pandas/series/#6
ticket_freq[ticket_freq > 3]


# In[7]:


ticket_freq[ticket_freq > 3].plot(figsize=(15,10), kind="bar")


# In[8]:


# total_data.groupby('Survived')['Ticket'].value_counts(dropna=True).unstack(0).plot.bar()
# total_data.groupby('Ticket')['Survived'].value_counts(dropna=False).unstack().plot.bar(stacked=True)
total_data_ticket = total_data.groupby("Ticket")["Survived"].value_counts(dropna=False).unstack()
total_data_ticket.fillna(0, inplace=True)
total_data_ticket.columns = ["nan", "d", "s"]
total_data_ticket["count"] = total_data_ticket.sum(axis=1)
total_data_ticket[total_data_ticket["count"] > 3].sort_values("count", ascending=False)[["nan", "d", "s"]].plot.bar(figsize=(15,10),stacked=True)
plt.savefig("ticket.svg")

# https://own-search-and-study.xyz/2016/08/03/pandas%E3%81%AEplot%E3%81%AE%E5%85%A8%E5%BC%95%E6%95%B0%E3%82%92%E4%BD%BF%E3%81%84%E3%81%93%E3%81%AA%E3%81%99/
# ax = total_data_ticket[total_data_ticket["count"] > 3].sort_values("count", ascending=False)[["nan", "d", "s"]].plot.bar(figsize=(15,10),stacked=True)
# fig = ax.get_figure()
# fig.savefig('ticket.png')


# In[9]:


total_data[total_data["Ticket"]=="CA. 2343"]


# In[10]:


total_data[total_data["Ticket"]=="1601"]


# In[11]:


total_data[total_data["Ticket"]=="CA 2144"]


# In[12]:


total_data[total_data["Ticket"]=="S.O.C. 14879"]


# In[13]:


total_data[total_data["Ticket"]=="347082"]


# In[ ]:




