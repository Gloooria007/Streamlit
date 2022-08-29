#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries
import streamlit as st
import os

# User Parameters
WORKDIR = 'D:/Python/jupyter/0.capstone/streamlit/'

# Main
def main():
    # Header of App
    st.title("Predict Suicidal Risk based on Social Media Posts")

    # Add background image
    # set_background(MAIN_BG)

    # Menu
    st.sidebar.text("Streamlit version: " + str(st.__version__))
    menu = ["About","Predict"]
    choice = st.sidebar.selectbox("Section:", menu)

    if choice == "About":
        exec(open(WORKDIR + 'webpage_about.py').read())

    elif choice == "Predict":  
        exec(open(WORKDIR + 'webpage_pred.py').read())

if __name__ == '__main__':
    main()

