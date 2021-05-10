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

choose_option = st.selectbox(
    "Choose a Task",
    [
        " ",
        "1. Upload & Display Raw Data",
        "2. Create Data Bar Chart",
        "3. Train Model and Predict"
    ]
    )

def upload_file():
    uploaded_file = st.file_uploader("", type = "csv")
    if uploaded_file is None:
        st.stop()
    if uploaded_file is not None:
        df=pd.read_csv(uploaded_file)
        df = pd.DataFrame(df)
    return df

def bar_chart(df):
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
    plt.title("Pilot Attributes Bar Chart", fontsize=16)
    return fig

def ml(df):
    y=df["# of Retrains"]
    X = df.drop("# of Retrains", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, random_state=42, test_size = 0.2)
    estimator = gbr()
    estimator.fit(X_train, y_train)
    return estimator

#creates a sidebar slider
def get_user_input(df):
    value = []
    names = df.columns
    dictionary = {}
    #iteration to create the sliders
    for name in names:
        minimum = 0 #as item because of the int32, int mismatch
        maximum = 1
        sliders = st.sidebar.slider(str(name), minimum, maximum, 1)
        value.append(sliders)
    #return value
    dictionary = dict(zip(names, value))
    features = pd.DataFrame(dictionary, index=[0])
    return features

def main():
    if choose_option == "1. Upload & Display Raw Data":
        df = upload_file()
        st.write(df.head())
        
        return df

    if choose_option == "2. Create Data Bar Chart":
        df = upload_file()
        fig = bar_chart(df)
        st.pyplot(fig)

    if choose_option == "3. Train Model and Predict":
        df=upload_file()
        estimator = ml(df)
        features = get_user_input(df.drop(columns = "# of Retrains"))
        st.write(features)
        output = np.round(estimator.predict(features))
        st.write(f" The prediction for this student is {abs(output.item()):.0f} retrain(s)")


if __name__ == "__main__":
    main()