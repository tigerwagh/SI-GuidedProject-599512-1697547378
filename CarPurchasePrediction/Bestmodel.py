
import numpy as np
import pandas as pd

df = pd.read_csv('car_data.csv')
df.head()

df.shape

df.Purchased.value_counts()

df.isnull().sum()

df = df.drop(columns = ['User ID'],axis = 1)

df.head()

from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()

df.Gender = le.fit_transform(df.Gender)

df.head()

import matplotlib.pyplot as plt
import seaborn as sns

sns.displot(df.Age)

df.describe()

sns.boxplot(df.AnnualSalary)

sns.boxplot(df.Age)

df.corr()

sns.heatmap(df.corr(),annot =True)

df.corr().Purchased.sort_values(ascending=False)

df.head()

## X and y split

X =df.drop(columns =['Purchased'],axis =1)

X.head()

y =df.Purchased
y.head()

from sklearn.preprocessing import MinMaxScaler
scale =MinMaxScaler()

scaled_x = pd.DataFrame(scale.fit_transform(X),columns =X.columns)
scaled_x.head()

# Train test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(scaled_x,y,test_size = 0.2,random_state = 0)

x_train.shape

x_test.shape

## Model building
from sklearn.tree import DecisionTreeClassifier
model1=DecisionTreeClassifier(max_depth=4,splitter='best',criterion='entropy')
model1.fit(x_train,y_train)
y_predict_1=model1.predict(x_test)
y_predict_1
y_predict_train=model1.predict(x_train)

import pickle

pickle.dump(model1,open('car_.pkl','wb'))