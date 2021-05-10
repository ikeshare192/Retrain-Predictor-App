import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skewnorm
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
import streamlit as st
from io import StringIO
import time

st.image('INDEX2.jpg', width=700)
#The Header
st.markdown("<h1 style='text-align: center; color: blue; '>RETRAIN PREDICTION APP</h1>" \
    , unsafe_allow_html=True)

st.subheader("READ ME:")
st.write("The pupose of this application is to demonstrate the value of Machine Learning \
    used as a data analytics tool that enhaces data driven decision making.  This model will \
    validate the process of determining the potential number of retrains a new student will \
    have by taking in a dataframe that resembles the traits of an organization of professional \
    pilots.  The features currently include 13 professional traits and the target variable is the \
    number of retrains.  The output of this model is a single prediction of  retrains or batch \
    predictions for a class.")

st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Choose a csv file", type="csv")

#This causes the script to stop and not throw an error flag after initiation.
if uploaded_file is None:
    st.stop()

if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    encoding = 'utf-8'
    s = str(bytes_data, encoding)
    data = StringIO(s)
    df = pd.read_csv(data, index_col=False)
    st.write(df.head(3))
    st.text(f"Your dataset contains {len(df)} rows and {len(df.columns)} columns")

#This calculates the sum per column.  This gets used in the bar chart plot script
x=np.arange(len(df.columns))
col_sums = []
for column in df.columns:
    sums = sum(df[column])
    col_sums.append(sums)

#creating the bar chart script
ys = np.arange(len(df.columns))
x_labels = df.columns
my_colors = ["red","green","black","blue","yellow", "magenta"]

fig, ax = plt.subplots(figsize = (12,10))
plt.bar(ys,col_sums,color = my_colors)

plt.xticks(ys, x_labels,rotation=90)
plt.xlabel('Features')
plt.ylabel('Model Attribute(sums)')
plt.title("Pilot Attributes Bar Chart", fontsize=16)
st.pyplot(fig)

y=df["# of Retrains"]
X = df.drop("# of Retrains", axis=1)

#splitting the train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,y, random_state=42, test_size = 0.2)

#creating the model as a variable
estimator = gbr()

#fitting the model
estimator.fit(X_train, y_train)

st.write(estimator.score(X_test, y_test))