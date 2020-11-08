# show_data.py ; see csv data
# to run .py in ipython, %run .py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data.describe()
test_data.describe()

# https://www.codexa.net/kaggle-titanic-beginner/
def kesson_table(df):
        null_val = df.isnull().sum()
        percent = 100 * null_val / len(df)
        kesson_table = pd.concat([null_val, percent], axis=1)
        kesson_table_ren_columns = kesson_table.rename(
        columns = {0 : '欠損数', 1 : '%'})
        return kesson_table_ren_columns

# kesson_table(train_data)
# kesson_table(test_data)

# Sex
train_data_male = train_data[train_data["Sex"] == "male"]
train_data_female = train_data[train_data["Sex"] == "female"]
male_female_survived = pd.DataFrame(np.zeros((2, 2)),
    columns=['Survived', 'Not survived'],
    index=['Male', 'Female'])
male_female_survived.loc['Male', 'Survived'] = len(train_data_male[train_data_male["Survived"]==1])
male_female_survived.loc['Male', 'Not survived'] = len(train_data_male[train_data_male["Survived"]==0])
male_female_survived.loc['Female', 'Survived'] = len(train_data_female[train_data_female["Survived"]==1])
male_female_survived.loc['Female', 'Not survived'] = len(train_data_female[train_data_female["Survived"]==0])
# https://qiita.com/ynakayama/items/9979258ac68cb669757a
# male_female_survived.plot(kind='bar', stacked=True)
# plt.grid()
# plt.savefig("sex.svg")

# Age
male0 = train_data_male[train_data_male["Survived"]==0]
male1 = train_data_male[train_data_male["Survived"]==1]
female0 = train_data_female[train_data_female["Survived"]==0]
female1 = train_data_female[train_data_female["Survived"]==1]

figure, (Left, Right) = plt.subplots(ncols=2, figsize=(10, 5))

Left.hist([male1['Age'].dropna(), male0['Age'].dropna()], label=['Survived', 'Not survived'],
    stacked=True)
Left.legend()
Left.set_xlim(0, 100)
Left.set_ylim(0, 120)
Left.grid(True)
Left.set_title("Male")
Left.set_xlabel("Age")

Right.hist([female1['Age'].dropna(), female0['Age'].dropna()], label=['Survived', 'Not survived'],
    stacked=True)
Right.legend()
Right.set_xlim(0, 100)
Right.set_ylim(0, 120)
Right.grid(True)
Right.set_title("Female")
Right.set_xlabel("Age")

figure.savefig("age.svg")
