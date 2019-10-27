#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RANSACRegressor
import seaborn as sns
import scipy as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
import tensorflow.contrib.keras as keras
import tensorflow as tf
import keras
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
#np.set_printoptions(suppress=True)
#np.set_printoptions(np.inf)
import matplotlib.pyplot as pl
import statsmodels.api as sm
pd.set_option('display.max_rows', 500)
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns', 500)
train=pd.read_csv("C:/Users/Arata Yamamoto/Documents/データ/train.csv")
test=pd.read_csv("C:/Users/Arata Yamamoto/Documents/データ/test.csv")
train.head()


# In[10]:


train["GarageYrBlt"].value_counts()
print(train.isnull().sum())


# In[12]:


#欠損地処理
train=pd.get_dummies(train,dummy_na=True)
train['LotFrontage']=train['LotFrontage'].interpolate(method='linear', limit_direction='forward', limit_area='inside')
train['GarageYrBlt_dummy']=train["GarageYrBlt"].apply(lambda x: 0 if type(x) == float else 1)
train['MasVnrArea_dummy']=train["MasVnrArea"].apply(lambda x: 0 if type(x) == float else 1)
train.head()
print(train.isnull().sum())


# In[11]:


plt.hist(train['GrLivArea'])


# In[22]:


y=train['SalePrice'].values
X=train.drop(['SalePrice','GarageYrBlt','MasVnrArea','Id'],axis=1).values
X_column=train.drop(['SalePrice','GarageYrBlt','MasVnrArea','Id'],axis=1)
columns=X_column.columns
(X_train, X_test,
 y_train, y_test) = train_test_split(
    X, y, test_size=0.3, random_state=0,
)
X_corr=train.drop(['SalePrice','GarageYrBlt','MasVnrArea','Id'],axis=1)
heatmap=X_corr.corr()
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 16))
sns.heatmap(heatmap, vmax=1, vmin=-1, center=0)


# In[25]:


X_corr2=train.iloc[:,1:10]
heatmap2=X_corr2.corr()
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 16))
sns.heatmap(heatmap2, vmax=1, vmin=-1, center=0)


# In[26]:


pg = sns.pairplot(X_corr2)
print(type(pg))


# In[5]:


lr=LinearRegression(normalize=False)

lr.fit(X_train,y_train)
lr.score(X_train,y_train)

# matplotlib パッケージを読み込み
import matplotlib.pyplot as pl

y_train_pred=lr.predict(X_train)
y_test_pred=lr.predict(X_test)
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# In[36]:


import statsmodels.api as sm
logit = sm.OLS(y_train, X_train)
X_train=sm.add_constant(X_train)
result = logit.fit()
result.summary2()


# In[28]:


lasso = Lasso(alpha=1000)
lasso.fit(X_train, y_train)
y_train_pred=lasso.predict(X_train)
y_test_pred=lasso.predict(X_test)
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

y_correct=y_test-y_test_pred
print(y_correct.mean())


# In[18]:


#欠損値処理
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
kesson_1=["Alley","GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]
kesson_2=["LotFrontage","Alley","GarageType","GarageYrBlt","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]
for i in kesson_1:
    train[i] = le.fit_transform(train[i].astype(str))

for i in kesson_2:
    train[i]=train[i].fillna(train[i].median())


# In[ ]:





# In[19]:


train["GarageType2"] = le.fit_transform(train['GarageType'].astype(str))
train["GarageType2"].value_counts()


# In[29]:


train=pd.get_dummies(train,dummy_na=True)
train.head()


# In[25]:


#変数作成
train_dummy=train.drop(["LotFrontage","Alley","LotArea","YearBuilt","YearRemodAdd","MasVnrType","MasVnrArea","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","Fireplaces","FireplaceQu","GarageType","GarageYrBlt","GarageFinish","GarageCars","GarageArea" ,"GarageQual" ,"GarageCond","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","PoolQC","Fence","MiscFeature","MiscVal","MoSold","YrSold"], axis=1)

train_dummy=pd.get_dummies(train_dummy)
train_missingnumeric=train[["Id","LotFrontage","Alley","LotArea","YearBuilt","YearRemodAdd","MasVnrType","MasVnrArea","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","Fireplaces","FireplaceQu","GarageType","GarageYrBlt","GarageFinish","GarageCars","GarageArea" ,"GarageQual" ,"GarageCond","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","PoolQC","Fence","MiscFeature","MiscVal","MoSold","YrSold"]]

train=pd.get_dummies(train)
train.head()


# In[127]:


train=train.dropna()
x = train.drop(['Id','SalePrice'], axis=1)
x_train=x.values
y_train=train['SalePrice'].values
lr=linear_model.LinearRegression()

lr.fit(x_train,y_train)
print(lr.coef_)
print(lr.score(x_train, y_train))
#print("coefficient = ", lr.coef_)
'''print("intercept = ", lr.intercept_)
param=lr.intercept_
param=param.ravel()

param2=lr.coef_
param2=param2.ravel()
param2

param2=np.append(param2,param)

param2'''


# In[120]:


train=train.dropna()
X = train.drop(['Id','SalePrice'], axis=1)
X = sm.add_constant(X)
Y = train['SalePrice']
model = sm.OLS(Y, X)
result = model.fit()
result.summary()


# In[5]:





# In[ ]:




