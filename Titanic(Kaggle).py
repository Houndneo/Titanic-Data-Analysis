#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split


# In[11]:


train_data=pd.read_csv("train (2).csv")
train_data


# In[12]:


train_data.shape


# In[13]:


train_data.head()


# In[14]:


train_data.info()


# In[15]:


test_data=pd.read_csv("test (1).csv")
test_data


# In[16]:


test_data.info()


# In[17]:


train_data['Survived'].value_counts()


# In[18]:


train_data['SibSp'].value_counts()


# In[19]:


#EXPLORING NUMERICAL DATA
train_data.describe()


# In[20]:


train_data.corr()


# In[21]:


train_data.groupby('Pclass')['Fare'].describe()


# In[23]:


from scipy import stats
def describe_cont_feature(feature):
    print(f'\n *** Results{feature}***')
    print(train_data.groupby('Survived')[feature].describe())
    print(ttest(feature))
def ttest(feature):
    survived=train_data[train_data['Survived']==1][feature]
    not_survived=train_data[train_data['Survived']==0][feature]
    tstat,pval=stats.ttest_ind(survived,not_survived,equal_var=False)
    print(f't-statistic:{tstat},p-value:{pval}')


# In[24]:


for feature in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
    describe_cont_feature(feature)


# In[25]:


train_data.groupby(train_data["Age"].isnull()).mean()


# In[26]:


train_data.describe(include='O')


# In[27]:


train_data.isnull().sum()


# In[28]:


test_data.isnull().sum()


# In[29]:


all_df = [train_data, test_data]
for i in all_df:
    mean = i['Age'].mean()
    std = i['Age'].std()
    is_null = i['Age'].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    age = i['Age'].copy()
    age[np.isnan(age)] = rand_age
    i['Age'] = age
    i['Age'] = i['Age'].astype(int)


# In[30]:


train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])


# In[31]:


import re
def get_title(name):
    find_title = re.search('([A-Z][a-z]+)\.', name)
    if find_title:
        return find_title.group(1)
    else:
        return ''

#Replace 'Name' column by 'Title' column.
train_data['Title'] = train_data['Name'].apply(get_title)
train_data = train_data.drop('Name', axis = 1)
test_data['Title'] = test_data['Name'].apply(get_title)
test_data = test_data.drop('Name', axis = 1)


# In[32]:


train_data[['Title','Sex']].value_counts()


# In[33]:


title_df = train_data[['Title', 'Survived']].groupby('Title').mean()


# In[34]:


title_dict = {}
for i in range(len(list(title_df.index))):
    title_dict[list(title_df.index)[i]] = title_df.iloc[i,0]
title_dict['Dona']=0

all_df = [train_data, test_data]
for data in all_df:
    data['Title']=data['Title'].replace(title_dict)


# In[35]:


mean = test_data.loc[(test_data['Pclass']==3)][['Fare']].mean()[0]
test_data['Fare'] = test_data['Fare'].fillna(value=mean)


# In[36]:


train_data= train_data.drop(['PassengerId','Ticket','Cabin'], axis=1)
test_data = test_data.drop(['PassengerId','Ticket','Cabin'], axis=1)
train_data.head()


# In[37]:


train_data['Embarked'] = train_data['Embarked'].astype('category').cat.codes
test_data['Embarked'] = test_data['Embarked'].astype('category').cat.codes
train_data['Sex'] = train_data['Sex'].astype('category').cat.codes
test_data['Sex'] = test_data['Sex'].astype('category').cat.codes


# In[38]:


X_train = train_data.drop("Survived", axis=1)
y_train = train_data["Survived"]
X_test  = test_data


# In[39]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[40]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)


# In[41]:


from sklearn.metrics import accuracy_score
model.score(X_train, y_train)


# In[42]:


from sklearn.model_selection import GridSearchCV
parameters = {'kernel':['linear', 'poly', 'rbf'],'C':[0.1, 1, 10]}

from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
scorer = make_scorer(f1_score)

grid_obj = GridSearchCV(model, parameters, scoring=scorer)
grid_fit = grid_obj.fit(X_train, y_train)
best_svm = grid_fit.best_estimator_


# In[48]:


y_submission = best_svm.predict(X_test) 

