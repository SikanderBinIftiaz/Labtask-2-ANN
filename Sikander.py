#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv(r"C:\Users\24108139\Downloads\jroRco.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df=df.replace("?",np.nan)


# In[7]:


df


# In[8]:


df.duplicated().sum()


# In[9]:


df.drop_duplicates(inplace=True)


# In[10]:


df.duplicated().sum()


# In[11]:


cols=["workclass","occupation","native.country"]
for col in cols:
    df[col]=df[col].fillna(df[col].mode()[0])


# In[12]:


df.isnull().sum()


# In[13]:


from sklearn.preprocessing import LabelEncoder


# In[14]:


encoder=LabelEncoder()
cat_cols=["workclass","education","marital.status","occupation","relationship","race","native.country","sex"]
for col in cat_cols:
    df[col]=encoder.fit_transform(df[col])


# In[15]:


df.info()


# In[16]:


x=df.drop("income",axis=1)
y=df["income"]


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[19]:


from sklearn.linear_model import LogisticRegression


# In[20]:


model=LogisticRegression(max_iter=1000,random_state=42)


# In[23]:


model.fit(x_train,y_train)


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


y_pred=model.predict(x_test)


# In[30]:


from sklearn.metrics import accuracy_score


# In[33]:


print("The Accuracy of the Logistic Regression Model is:",accuracy_score(y_test,y_pred)*100,"%")


# In[ ]:




