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
train_data_num = len(train_data)
total_data = pd.concat([train_data, test_data])


# In[3]:


total_data.head()


# In[4]:


total_data.describe()


# In[5]:


total_data.describe(include="O")


# In[6]:


title_data = total_data["Name"].str.extract('(?P<Title>[A-Za-z]+)\.', expand=True)
title_data["Title"].value_counts().sort_values(ascending=True).plot.barh()

# https://mysuki.jp/english-honorific-25750


# In[7]:


title_data["Title"].value_counts()


# In[8]:


for t in ["Dr", "Rev", "Col", "Major", "Ms", "Mlle", "Countess", "Lady", "Sir", "Mme", "Capt", "Jonkheer", "Dona", "Don"]:
    title_data[title_data["Title"] == t] = "Else"
    

title_data["Title"].value_counts().sort_values(ascending=True).plot.barh()


# In[9]:


total_data = pd.concat([total_data, title_data], axis=1)
total_data.head()


# In[10]:


total_data.describe()


# In[11]:


total_data.describe(include="O")


# In[12]:


# 欠損値を埋める
# https://qiita.com/0NE_shoT_/items/8db6d909e8b48adcb203
total_data['Age'] = total_data['Age'].fillna(total_data['Age'].median())
total_data['Fare'] = total_data['Fare'].fillna(total_data['Fare'].median())
total_data['Embarked'] = total_data['Embarked'].fillna("S")


# In[13]:


total_data.describe()


# In[14]:


total_data.describe(include="O")


# In[15]:


# データを元通りに分割する
train_data = total_data[total_data['PassengerId'] <= train_data_num]
test_data = total_data[total_data['PassengerId'] > train_data_num]


# In[16]:


train_data.head()


# In[17]:


train_data.tail()


# In[18]:


test_data.head()


# In[19]:


test_data.tail()


# In[20]:


from sklearn.ensemble import RandomForestClassifier

features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Fare", "Embarked", "Title"]
y = train_data["Survived"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1, max_features="auto")
model.fit(X, y)
predictions = model.predict(X_test)


# In[21]:


print('Train score: {}'.format(model.score(X, y)))


# In[22]:


x_importance = pd.Series(data=model.feature_importances_, index=X.columns)
x_importance.sort_values(ascending=False).plot.bar()


# In[23]:


# グリッドサーチ
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": [2, 4, 6, 10, None],
              "n_estimators":[1, 3, 10, 30, 100, 300],
              "max_features":["auto", None, "log2"]}

model_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=1),
                 param_grid = param_grid,   
                 scoring="accuracy",  # metrics
                 cv = 3,              # cross-validation
                 n_jobs = 1)          # number of core

model_grid.fit(X, y) #fit

model_grid_best = model_grid.best_estimator_ # best estimator
print("Best Model Parameter: ", model_grid.best_params_)


# In[24]:


print('Train score: {}'.format(model_grid_best.score(X, y)))


# In[25]:


predictions = model_grid_best.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output = output.astype({'Survived' : 'int64'})
# output.info()
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[26]:


# remove Embarked
features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Fare", "Title"]
y = train_data["Survived"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1, max_features="auto")
model.fit(X, y)
predictions = model.predict(X_test)


# In[27]:


print('Train score: {}'.format(model.score(X, y)))


# In[28]:


x_importance = pd.Series(data=model.feature_importances_, index=X.columns)
x_importance.sort_values(ascending=False).plot.bar()


# In[29]:


# グリッドサーチ
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": [2, 4, 6, 10, None],
              "n_estimators":[1, 3, 10, 30, 100, 300],
              "max_features":["auto", None, "log2"]}

model_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=1),
                 param_grid = param_grid,   
                 scoring="accuracy",  # metrics
                 cv = 3,              # cross-validation
                 n_jobs = 1)          # number of core

model_grid.fit(X, y) #fit

model_grid_best = model_grid.best_estimator_ # best estimator
print("Best Model Parameter: ", model_grid.best_params_)


# In[30]:


print('Train score: {}'.format(model_grid_best.score(X, y)))


# In[31]:


predictions = model_grid_best.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output = output.astype({'Survived' : 'int64'})
# output.info()
output.to_csv('my_submission1.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




