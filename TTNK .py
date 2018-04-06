
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']


# In[2]:


data_train = pd.read_csv(r'C:\Users\WENNY CUNXI\Desktop\taitannike\train.csv')
data_train.info()


# In[24]:


data_train.describe()


# In[15]:


fig = plt.figure()

plt.subplot2grid((2, 3), (0, 0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u'获救情况')
plt.ylabel(u'人数')

plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.title(u'乘客等级分布')
plt.ylabel(u'人数')

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)
plt.title(u'年龄与获救情况')
plt.ylabel(u'年龄')

plt.subplot2grid((2, 3), (1, 0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.title(u'各船舱等级乘客年龄分布')
plt.xlabel(u'年龄')
plt.ylabel(u'密度')
plt.legend((u'头等舱', u'二等舱', u'三等舱'), loc='best')

plt.subplot2grid((2, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title('各登船口岸上岸人数')
plt.ylabel('人数')

plt.tight_layout()
plt.savefig(u'图像.png', dpi=400)


# In[18]:


fig1 = plt.figure()

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = DataFrame({u'未获救' : Survived_0, u'获救' : Survived_1})
df.plot(kind='bar', stacked=True)
plt.title(u'各船舱等级获救情况')
plt.xlabel(u'乘客等级')
plt.ylabel(u'人数')
plt.legend((u'获救', u'未获救'), loc='best')

plt.savefig('各船舱获救等级.png', dpi=400)


# In[19]:


fig2 = plt.figure()

Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
df1 = DataFrame({u'女性':Survived_f, u'男性':Survived_m})
df1.plot(kind='bar', stacked=True)
plt.title(u'按性别看获救情况')
plt.xlabel(u'性别')
plt.ylabel(u'人数')
plt.legend((u'女性', u'男性'), loc='best')

plt.savefig(u'按性别看获救情况.png', dpi=400)


# In[33]:


fig3 = plt.figure(figsize=(15, 5))
plt.title(u'根据船舱等级看获救情况')

ax1 = fig3.add_subplot(141)
data_train.Survived[data_train.Sex =='female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='lady high class', color='r')
ax1.set_xticklabels([u'获救', u'未获救'], rotation=0)
ax1.legend((u'高级舱'), loc='best')
plt.xlabel(u'女性')

ax2 = fig3.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='lady low class', color='y')
ax2.set_xticklabels([u'获救', u'未获救'], rotation=0)
ax2.legend((u'低级舱'), loc='best')
plt.xlabel(u'女性')

ax3 = fig3.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='man high class', color='b')
ax3.set_xticklabels([u'获救', u'未获救'], rotation=0)
ax3.legend((u'高级舱'), loc='best')
plt.xlabel(u'男性')

ax4 = fig3.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='man low class', color='g')
ax4.set_xticklabels([u'获救', u'未获救'], rotation=0)
ax4.legend((u'低级舱'), loc='best')
plt.xlabel(u'男性')


plt.savefig('性别、船舱和获救情况.png', dpi=400)


# In[34]:


fig4 = plt.figure()

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u'各登录港口乘客的获救情况')
plt.xlabel(u'登录港口')
plt.ylabel(u'人数')

plt.savefig('登录港口与获救情况.png', dpi=400)


# In[36]:


g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print (df)


# In[37]:


g = data_train.groupby(['Parch','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print (df)


# In[38]:


data_train.Cabin.value_counts()


# In[3]:


from sklearn.ensemble import RandomForestRegressor


# In[5]:


def set_missing_ages(df):
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    
    know_age = age_df[age_df.Age.notnull()].as_matrix()
    unknow_age = age_df[age_df.Age.isnull()].as_matrix()
    
    y = know_age[:, 0]
    X = know_age[:, 1:]
    
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    
    predictedAges = rfr.predict(unknow_age[:, 1::])
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges
    
    return df, rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'YES'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'NO'
    return df
    
data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)


# In[6]:


data_train


# In[7]:


dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')

df = pd.concat([data_train, dummies_Cabin, dummies_Pclass, dummies_Sex, dummies_Embarked], axis=1)
df.drop(['Name', 'Cabin', 'Pclass', 'Sex', 'Embarked', 'Ticket'], axis=1, inplace=True)
df


# In[10]:


import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
scale_param = scaler.fit(df[['Age', 'Fare']])
df['Age_scaled'] = scaler.fit_transform(df[['Age', 'Fare']], scale_param)[:, 0]
df['Fare_scaled'] = scaler.fit_transform(df[['Age', 'Fare']], scale_param)[:, 1]
df


# In[11]:


import re


# In[13]:


from sklearn import linear_model
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()
y = train_np[:, 0]
X = train_np[:, 1:]
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)
clf


# In[14]:


data_test = pd.read_csv(r'C:\Users\WENNY CUNXI\Desktop\taitannike\test.csv')
data_test.info()


# In[32]:


data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges


# In[24]:


data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
scaler = preprocessing.StandardScaler()
scale_param = scaler.fit(df_test[['Age', 'Fare']])
df_test['Age_scaled'] = scaler.fit_transform(df_test[['Age', 'Fare']], scale_param)[:, 0]
df_test['Fare_scaled'] = scaler.fit_transform(df_test[['Age', 'Fare']], scale_param)[:,1]

df_test


# In[33]:


test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
test_np = df_test.as_matrix()

predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv(r'C:\Users\WENNY CUNXI\Desktop\taitannike\ttnkresult.csv', index=False)

