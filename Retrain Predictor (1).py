#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skewnorm
import time


# #### Below begins the building of the dataset. It starts with creating a column called milcol.  Its values are auto-generated with numpy's random integer generated and will be a 1 or a 0.

# In[ ]:


#creating the military data column.
milcol = list(np.random.randint(0,2,(12000)))


# #### The civilian data column is built by iterating about the milcol list.  Any value of 1 will be converted to a 0 and vice versa.

# In[ ]:


#creating the civilian data column.
civcol=[]
for i in milcol:
    if i.item() == 1:
        civcol.append(i.item()-1)
    if i.item() == 0:
        civcol.append(i.item()+1)


# #### The rest of the data set is populated with random integers and then everything is combined into one dataframe.

# In[ ]:


#creating a variable with all column names
column_names=['Flew Fighter','Flew Cargo','Flew Corporate',
              'Flew RJ', 'Arts Degree','Stem Degree',
              'A-Hours','B-Hours','C-Hours',
              'Prev Capt','Prev Rot-Wing']

#creating a synthetic dataset of random numbers either 0 or 1
syn_data =pd.DataFrame(np.random.randint(0,2,(12000,11)),
                       columns = column_names)

#creating a synthetic dataset of military and civilian columns
df1=pd.DataFrame(
    {"Is Military":milcol,
     "Is Civilian":civcol}
               )


# In[ ]:


#combining the two datasets above to create one
df2 = pd.concat([df1, syn_data], axis=1)
df2


# #### Take a look at the df_final dataframe above and make sure it is fully popluated.  There are currently some value combinations along the rows that are highly unlikely and need to be addressed.  Below is the code that creates conditions where each row will make sense and replicate the traits of a real pilot.  i.e. No civilians that flew fighter jets, etc., etc.

# In[ ]:


#if Is Civilian, then Flew Fighters = 0
ftr_is_zero = (df2["Is Civilian"]==1)
df2.loc[(ftr_is_zero, "Flew Fighter")]=0

#if Flew Fighter = 1, then Flew Cargo = 0
trans_is_zero = (df2["Flew Fighter"]==1)
df2.loc[(trans_is_zero,"Flew Cargo")]=0

#if Flew Fighter = 1, then Flew Corporate = 0
no_corp_if_ftr = (df2["Flew Fighter"]==1)
df2.loc[no_corp_if_ftr, "Flew Corporate"]=0

#if Flew Cargo = 1, then Flew RJ = 0
no_car_rj = (df2["Flew Cargo"]==1)
df2.loc[no_car_rj, "Flew RJ"]=0

#if Arts Degree = 1, then Stem Degree = 0
other_deg = (df2["Arts Degree"]==1)
df2.loc[(other_deg, "Stem Degree")]=0

#if Arts Degree = 0, then Stem Degree = 1
other_deg2 = (df2["Arts Degree"]==0)
df2.loc[(other_deg2, "Stem Degree")]=1

#if Is Military = 1, then Prev Capt = 0
never_cpt = (df2["Is Military"]==1)
df2.loc[never_cpt, "Prev Capt"]=0

#if Flew Corporate = 1, then Flew RJ = 0
not_rj = (df2["Flew Corporate"]==1)
df2.loc[not_rj, "Flew RJ"]=0

#if A-Hours = 1, then B-Hours = 0
hours1 = (df2["A-Hours"]==1)
df2.loc[(hours1, "B-Hours")]=0

#if B-Hours = 1, then C-Hours = 0
hours2 = (df2["B-Hours"]==1)
df2.loc[(hours2, "C-Hours")]=0

#if A-Hours = 1, then C-Hours = 0
hours3 = (df2["A-Hours"]==1)
df2.loc[(hours3, "C-Hours")]=0

#if Is Civilian = 1, then Prev Rot-Wing = 0
no_helo1 = (df2["Is Civilian"]==1)
df2.loc[(no_helo1, "Prev Rot-Wing")]=0

#if Flew Fighter = 1, then Prev Rot-Wing = 0
no_helo2 = (df2["Flew Fighter"]==1)
df2.loc[(no_helo2, "Prev Rot-Wing")]=0          


# In[ ]:


#A more accuract desctiption of a poplution of pilots
df2


# In[ ]:


#low time helo to fixed wing transition array
helo = np.array([[1,0,0,0,1,0,1,0,1,0,0,0,1]])

#low time fighter pilot array
low_time = np.array([[1,0,1,0,0,0,0,1,1,0,0,0,0]])


# In[ ]:


#creating bias data
helo_bias = pd.DataFrame(helo, columns = df2.columns)
low_time_bias = pd.DataFrame(low_time, columns = df2.columns)

#populating a helo bias table
helo_bias = pd.concat([helo_bias]*2000, axis=0, ignore_index=True)
#populating a low time bias table
low_ti_bias = pd.concat([low_time_bias]*3000, axis=0, ignore_index=True)

#combining low time bias and helo bias dataframes into one 
df_bias = pd.concat([helo_bias, low_ti_bias], axis=0)

#Creating the final data frame which is a combination of#
#the synthetic data and the bias data####################
final = pd.concat([df2, df_bias], ignore_index=True)


# In[ ]:


#quick view of the final dataframe
final


# In[ ]:


#creating random variables for the retrain targets
norm_ret = pd.DataFrame(np.random.randint(0,1,(12000)))

#create a biased retrain target away from the normal mean
bias_ret = pd.DataFrame(np.random.randint(1,3,(5000)))


# In[ ]:


final["# of Retrains"] = pd.concat([norm_ret,bias_ret], ignore_index=True)
final


# #### The section below begins the process of plotting, and processing the data through ML algorithms.

# In[ ]:


#This cell adds the values in every column and creates a list#
#of sums.  Those values will be used in the bar charting.#####

x=np.arange(len(final.columns))
col_sums = []
for column in final.columns:
    sums = sum(final[column])
    col_sums.append(sums)
    
#view the sum of each column as a list
final.columns,col_sums


# In[ ]:


#ax=sns.barplot(x=np.arange(len(df_final.columns)), y=col_sums)
#(df_final.columns)


ys = np.arange(len(final.columns))
x_labels = final.columns
my_colors = ["red","green","black","blue","yellow", "magenta"]

fig, ax = plt.subplots(figsize = (12,10))
plt.bar(ys,col_sums,color = my_colors)

plt.xticks(ys, x_labels,rotation=90)
plt.xlabel('Features')
plt.ylabel('Model Attribute(sums)')
plt.title("Model Attributes", fontsize=16)


# #### Ultimately our real data set would have move variability in it but for a snthetic data set this is a great start.  Below we need to create a data set that is our target variable.  That variable will be a number of retrains that have historically occured with the given pilot attributes.  The goal is to train an algorithm that will be able to predict a number of retrains given a pilots attributes.

# In[67]:


#plotting a histogram of the retrain column
fig, ax = plt.subplots(figsize = (12,8))
sns.histplot(
    final["# of Retrains"],
    bins=3,
    color = 'green',
    kde = False)

legends = ['##']
plt.legend(legends)
ax.set_title("Histogram of retrains")
ax.set_xlabel("# of Retrains")
ax.set_ylabel("Frequency")


# In[ ]:


#labeling the X(features) and y(targets)
y=final["# of Retrains"].astype(int)
X=final.drop(("# of Retrains"), axis=1)


# In[ ]:


y


# In[ ]:


X


# In[ ]:


#importing the necessary libraries to run the ML algorithms
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split


# In[ ]:


#splitting the train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,y, random_state=42, test_size = 0.2)


# In[ ]:


estimator = gbr()


# In[ ]:


help(LR)


# In[31]:


get_ipython().run_cell_magic('time', '', 'estimator.fit(X_train, y_train)')


# In[34]:


estimator.score(X_test, y_test)


# In[35]:


estimator.feature_importances_


# In[36]:


y_pos = np.arange(len(list(X.columns)))
bar_labels = X.columns
coef = estimator.feature_importances_
coefs = [i.tolist() for i in coef]


fig, ax = plt.subplots(figsize = (10,10))
my_colors2 = ["green","blue","orange","red","yellow","black","magenta"]
plt.bar(y_pos,coef, color = my_colors2 )
        #[val for sublist in coefs for val in sublist])
plt.xticks(y_pos, bar_labels, rotation=90)
plt.xlabel('Features')
plt.ylabel('Model Coefficients')
coefs


# In[37]:


pilot_newbie = np.array([1,0,0,1,0,0,1,0,0,0,1,0,0]).reshape(1,-1)
newbie_prediction = estimator.predict(pilot_newbie)


# In[38]:


print(f"Your trainee is predicted to have {abs(np.round(newbie_prediction.item())):.0f} retrains")


# In[39]:


shareef_fng = np.array([1,0,1,0,0,0,0,0,1,0,0,0,0]).reshape(1,-1)
shareef_prediction  = estimator.predict(shareef_fng)


# In[40]:


print(f"Isaac Shareef will likely have {np.round(shareef_prediction.item()):.0f} retrain")


# In[41]:


class_a = X.sample(20,replace = True, random_state=42)


# In[42]:


class_a


# In[43]:


arr_cls_a = np.array(class_a)


# In[44]:


#creating an array of arrays
arr_cls_a


# In[45]:


rets = np.abs(np.round(estimator.predict(arr_cls_a)))


# In[46]:


print(f"This class is predicted to have a sum of {sum(rets):.0f} retrains")


# In[59]:


prob_est = LR(max_iter = 1000, random_state = 42)


# In[60]:


prob_est.fit(X_train, y_train)


# In[61]:


prob_est.score(X_test, y_test)


# In[62]:


prob_est.predict(pilot_newbie)


# In[63]:


prob_est.predict(shareef_fng)


# In[64]:


prob_est.predict_proba(shareef_fng)


# In[65]:


helo_boy = np.array([1,0,0,0,1,0,1,0,1,0,0,0,1]).reshape(1,-1)
prob_est.predict(helo_boy)


# In[66]:


prob_est.predict_proba(helo_boy)


# In[69]:


prob_est.predict_proba(pilot_newbie)


# In[70]:


prob_est.predict_proba(class_a)


# In[ ]:




