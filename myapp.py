#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


# In[2]:


st.write("Iris Flower Predition app: This app predit the iris flower type")


# In[3]:


st.sidebar.header("User Input Parameters")


# In[4]:


def user_input_features():
    sepal_lenght=st.sidebar.slider("sepal length",4.3,7.9,5.4)
    sepal_width=st.sidebar.slider("sepal_width",2.0,4.4,3.4)
    petal_lenght=st.sidebar.slider("petal lenght",1.0,6.9,1.3)
    petal_width=st.sidebar.slider("petal_width",0.1,2.5,0.2)
    data={"sepal_lenght":sepal_lenght,"sepal_width":sepal_width,"petal_lenght":petal_lenght,"petal_width":petal_width}
    features=pd.DataFrame(data,index=[0])
    return features


# In[5]:


df=user_input_features()
st.subheader("User Input Parameters")
st.write(df)


# In[6]:


iris = datasets.load_iris()
X=iris.data
Y=iris.target
clf=RandomForestClassifier()
clf.fit(X,Y)
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)


# In[7]:


st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)


# In[1]:


iris = datasets.load_iris()
X=iris.data
Y=iris.target
clf=RandomForestClassifier()
clf.fit(X,Y)
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)


# In[ ]:




