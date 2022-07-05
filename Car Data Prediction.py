#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("car data.csv")


# In[3]:


df.head()


# In[5]:


df.shape


# In[28]:


print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
print(df['Fuel_Type'].unique())


# In[13]:


###check misssing or null values
df.isnull().sum()


# In[14]:


df.describe()


# In[15]:


df.columns


# In[16]:


final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]


# In[17]:


final_dataset.head()


# In[18]:


final_dataset['Current_year']=2020


# In[19]:


final_dataset.head()


# In[20]:


final_dataset['no_year']=final_dataset['Current_year']-final_dataset['Year']


# In[21]:


final_dataset.head()


# In[22]:


final_dataset.drop(['Year'],axis=1,inplace=True)


# In[23]:


final_dataset.head()


# In[24]:


final_dataset.drop(['Current_year'],axis=1,inplace=True)


# In[25]:


final_dataset.head()


# In[26]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[27]:


final_dataset.head()


# In[29]:


final_dataset.corr()


# In[30]:


import seaborn as sns


# In[31]:


sns.pairplot(final_dataset)


# In[32]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


corrmat=final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap='coolwarm')


# In[38]:


final_dataset.head()


# In[48]:


###independent and dependent features
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[49]:


X.head()


# In[50]:


y.head()


# In[54]:


### Feature Importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)


# In[58]:


print(model.feature_importances_)


# In[60]:


##plot graph of features importance for better visualiztion
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[61]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[63]:


X_train


# In[64]:


X_train.shape


# In[67]:


from sklearn.ensemble import RandomForestRegressor
rf_random=RandomForestRegressor()


# In[70]:


###Hyperparamaters
import numpy as np
n_estimators=[int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[71]:


###Randomized Search CV

#Number of tree in rwndom forest
n_estimators=[int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
#Number of features to consider in every split
max_features = ['auto', 'sqrt']
#Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
#Max_depth.append(None)
#Minimum number of samples required to split a node
min_samples_split = [2,5,10,15,100]
#Minimum number of samples required to each leaf node
min_samples_leaf = [1,2,5,10]


# In[72]:


from sklearn.model_selection import RandomizedSearchCV


# In[73]:


#Create the random grid
random_grid = {'n_estimators' : n_estimators,
               'max_features' : max_features,
               'max_depth' : max_depth,
               'min_samples_split' : min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[74]:


## use the random grid to search for best hyperparameters
## First create the base model to tune
rf = RandomForestRegressor()


# In[78]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10,cv =5, verbose=2, random_state=42, n_jobs = 1)


# In[79]:


rf_random.fit(X_train,y_train)


# In[81]:


predictions=rf_random.predict(X_test)


# In[83]:


predictions


# In[84]:


sns.distplot(y_test-predictions)


# In[85]:


plt.scatter(y_test,predictions)


# In[86]:


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl','wb')

#dump information to that file
pickle.dump(rf_random, file)


# In[ ]:




