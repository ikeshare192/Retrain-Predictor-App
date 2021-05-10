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


#The Header
st.title("RETRAIN PREDICTION APP")
st.header("An Introductory Machine Learn App for UAL FTC")
st.write("The pupose of this application is to demonstrate the value of Machine Learning \
    used as a data analytics tool that enhaces data driven decision making.  This model will \
    validate the process of determining the potential number of retrains a new student will \
    have by taking in a dataframe that resembles the traits of an organization of professional \
    pilots.  The features will include 13 professional traits and the target will include the \
    number of retrains.  The output of this model is a single prediction or batch predictions \
    for a class.")

st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Choose a csv file", type="csv")
#bytes_data = uploaded_file.read()
#df = pd.DataFrame()
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    encoding = 'utf-8'
    s = str(bytes_data, encoding)
    data = StringIO(s)
    df = pd.read_csv(data, index_col=False)
    st.write(df.head(10))
    st.text(f"Your dataset contains {len(df)} rows and {len(df.columns)} columns")

x=np.arange(len(df.columns))
col_sums = []
for column in df.columns:
    sums = sum(df[column])
    col_sums.append(sums)

ys = np.arange(len(df.columns))
x_labels = df.columns
my_colors = ["red","green","black","blue","yellow", "magenta"]

fig, ax = plt.subplots(figsize = (12,10))
plt.bar(ys,col_sums,color = my_colors)

plt.xticks(ys, x_labels,rotation=90)
plt.xlabel('Features')
plt.ylabel('Model Attribute(sums)')
plt.title("Model Attributes Bar Chart", fontsize=16)
st.pyplot(fig)