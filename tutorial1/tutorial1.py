#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt


# In[2]:


train_data = pd.read_csv("../train.csv")
train_data.head()


# In[3]:


# test_data = pd.read_csv("../test.csv")
# test_data.head()


# In[4]:


train_data.info()


# In[5]:


train_data.describe()


# In[6]:


train_data.describe(include="O")


# In[7]:


train_data_map = train_data.copy()
train_data_map['Sex'] = train_data_map['Sex'].map({'male' : 0, 'female' : 1})
train_data_map['Embarked'] = train_data_map['Embarked'].map({'S' : 0, 'C' : 1, 'Q' : 2})
train_data_map_corr = train_data_map.corr()
train_data_map_corr


# In[8]:


from sklearn.model_selection import train_test_split

train_data_orig = train_data
train_data, cv_data = train_test_split(train_data_orig, test_size=0.3, random_state=1)


# In[9]:


train_data.info()


# In[10]:


cv_data.info()


# In[11]:


train_data.head()


# In[12]:


cv_data.head()


# In[13]:


from sklearn.ensemble import RandomForestClassifier

features = ["Pclass", "Sex", "SibSp", "Parch"]

y = train_data["Survived"]
y_cv = cv_data["Survived"]
X = pd.get_dummies(train_data[features])
X_cv = pd.get_dummies(cv_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1, max_features="auto")
model.fit(X, y)
predictions = model.predict(X_cv)


# In[14]:


print('Train score: {}'.format(model.score(X, y)))


# In[16]:


print('CV score: {}'.format(model.score(X_cv, y_cv)))


# In[27]:


rfc_results = pd.DataFrame(columns=["train", "cv"])


# In[49]:


for iter in [1, 10, 100]:
    model = RandomForestClassifier(n_estimators=iter, max_depth=4, random_state=1, max_features="auto")
    model.fit(X, y)
    predictions = model.predict(X_cv)
    rfc_results.loc[iter] = model.score(X, y), model.score(X_cv, y_cv)


# In[50]:


rfc_results


# In[55]:


from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": [2, 3, 4, 5, None],
              "n_estimators":[1, 3, 10, 30, 100],
              "max_features":["auto", None]}

model_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=1),
                 param_grid = param_grid,   
                 scoring="accuracy",  # metrics
                 cv = 3,              # cross-validation
                 n_jobs = 1)          # number of core

model_grid.fit(X, y) #fit

model_grid_best = model_grid.best_estimator_ # best estimator
print("Best Model Parameter: ", model_grid.best_params_)


# In[53]:


print('Train score: {}'.format(model.score(X, y)))


# In[54]:


print('CV score: {}'.format(model.score(X_cv, y_cv)))


# In[ ]:




