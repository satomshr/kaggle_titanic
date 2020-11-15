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


test_data = pd.read_csv("../test.csv")
test_data.head()


# In[4]:


def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
    columns = {0 : '欠損数', 1 : '%'})
    return kesson_table_ren_columns


# In[5]:


# kesson_table(train_data)


# In[6]:


# kesson_table(test_data)


# In[7]:


train_data.info()


# In[8]:


test_data.info()


# In[9]:


train_data.describe()


# In[10]:


train_data.describe(include="O")


# In[11]:


test_data.describe()


# In[12]:


test_data.describe(include="O")


# In[13]:


# https://qiita.com/ao_log/items/17b23da6a0b89c2866c8
train_data_map = train_data.copy()
train_data_map['Sex'] = train_data_map['Sex'].map({'male' : 0, 'female' : 1})
train_data_map['Embarked'] = train_data_map['Embarked'].map({'S' : 0, 'C' : 1, 'Q' : 2})
train_data_map_corr = train_data_map.corr()
train_data_map_corr


# In[14]:


seaborn.heatmap(train_data_map_corr, annot=True, vmax=1, vmin=-1, center=0)
plt.show()


# In[15]:


from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1, max_features="auto")
model.fit(X, y)
predictions = model.predict(X_test)

#train_data[features].head()
#X.head()


# In[16]:


print('Train score: {}'.format(model.score(X, y)))


# In[17]:


x_importance = pd.Series(data=model.feature_importances_, index=X.columns)
x_importance.sort_values(ascending=False).plot.bar()


# In[20]:


from dtreeviz.trees import dtreeviz
viz = dtreeviz(model.estimators_[0], X, y, target_name='Survived', feature_names=X.columns, class_names=['Not survived', 'Survived'])

viz.save("rfc.svg")
viz


# In[19]:


#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
#output.to_csv('my_submission.csv', index=False)
#print("Your submission was successfully saved!")

