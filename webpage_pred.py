#!/usr/bin/env python
# coding: utf-8

# In[2]:

# Import Libraries
import streamlit as st
import pickle
import base64
import pandas as pd
import numpy as np
from pipeline_text import run_pipeline as pipe1
from pipeline_csv import run_pipeline as pipe2
"""Webpage Content"""

st.write("""
This web app predicts the suicidal risk of a text!

The input format can either be a single **text** or a **csv file**.
""")

st.header("Input Text")

# Step 1: Ensure Column Name
st.markdown("""* **Step 1:** Input text """)
text = st.text_input("Input text")

# Step 2: Predict
st.markdown("""* **Step 2:** Predict the Suicidal Risk""")

pred_c1, pred_c2 = st.columns(2)
pred_but = pred_c1.button("Predict")

if pred_but:
        pred_c2.write(pipe1(text))

        

st.header("Input csv File")

# Step 1: Ensure Column Name
st.markdown("""* **Step 1:** Ensure **text input** are Defined with Specified Column Name: text" """)

# Step 2: Upload Input Data
st.markdown("""* **Step 2:** Upload Data in **CSV** Format""")
pred_datafile = st.file_uploader("", type=['CSV'])
if pred_datafile is not None:
    pred_dataset = pd.read_csv(pred_datafile)
    pred_dataset.drop(columns=["Unnamed: 0"], inplace=True)
    # check column name
    if 'text' not in pred_dataset.columns: 
        st.error('Warnings: "text" column is not found in uploaded csv file!')

# Step 3: Predict
st.markdown("""* **Step 3:** Download Prediction""")

if pred_datafile is not None:
    pred_dataset["Prediction"] = pipe2(pred_dataset)

    # Display Some Results
    st.markdown("*Display Some Results*")
    st.dataframe(pred_dataset.head())

    b64 = base64.b64encode(pred_dataset.to_csv(index=False).encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download Prediction</a> (Right-click and save as <filename>.csv)'
    st.markdown(href, unsafe_allow_html=True)

