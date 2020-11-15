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


total_data.head()


# In[4]:


total_data.describe()


# In[5]:


total_data.describe(include="O")


# In[16]:


title_data = total_data["Name"].str.extract('(?P<Title>[A-Za-z]+)\.', expand=True)
title_data["Title"].value_counts().sort_values(ascending=True).plot.barh()

# https://mysuki.jp/english-honorific-25750


# In[34]:


title_data["Title"].value_counts()


# In[7]:


total_data = pd.concat([total_data, title_data], axis=1)
total_data.head()


# In[25]:


master_data = total_data[total_data["Title"] == "Master"]
master_data.describe()

# "Master" は, 15 歳未満の男性
# https://qiita.com/hrappuccino/items/16c20645c60578391017
# master_data.groupby('Survived')['Age'].plot.hist(bins=20, alpha=0.5, legend=True)


# In[26]:


master_data.describe(include="O")


# In[27]:


miss_data = total_data[total_data["Title"] == "Miss"]
miss_data.describe()
# Miss は, 高齢者もいる. 女性のみ


# In[28]:


miss_data.describe(include="O")


# In[29]:


dr_data = total_data[total_data["Title"] == "Dr"]
dr_data.describe()


# In[30]:


dr_data.describe(include="O")


# In[31]:


dr_data.head(8)
# Age 不明者が 1 名


# In[35]:


rev_data = total_data[total_data["Title"] == "Rev"]
rev_data.describe()
# Rev ; 牧師


# In[36]:


rev_data.describe(include="O")


# In[37]:


rev_data.head(8)
# Rev は，生死が分かっている 6 名全員が死亡


# In[39]:


col_data = total_data[total_data["Title"] == "Col"]
col_data.head()
# Col ; 大佐


# In[40]:


major_data = total_data[total_data["Title"] == "Major"]
major_data.head()


# In[ ]:




