#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.datasets import load_iris
data = load_iris()


# In[6]:


data


# In[7]:


df = pd.DataFrame(data.data, columns= data.feature_names)
df


# In[9]:


df["target"] = data.target


# In[10]:


x=df.drop("target",axis="columns")
y=df["target"]


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


# In[12]:


len(x_train)


# In[13]:


len(y_test)


# In[14]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10) # the number of decision trees 
model.fit(x_train, y_train)
model.score(x_test, y_test)


# In[15]:


y_pr=model.predict(x_test)


# In[16]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pr)
cm


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




