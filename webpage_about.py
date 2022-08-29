"""Webpage of About
This page includes the motivation of the project, model performance and methodology.
Please add in necessary information
"""

# Import Libraries
import streamlit as st
from PIL import Image
#from Pillow import Image


# User Parameters
MODELFULL_PERF_FIG = 'D:/Python/jupyter/0.capstone/streamlit/Figure/model_perf.jpg'
STUDY_DSG = 'D:/Python/jupyter/0.capstone/streamlit/Figure/project_pipeline.jpg'


"""Webpage Content"""

# Motivation
st.header("Motivation")
st.markdown("""Suicide is the 10th leading leading cause of death worldwide.
COVID-19 has brought a lot of change and uncertainties in people's lives. 
The overall economic pressure, loss of a job, lack of physical contact with 
family and friends, the panic of uncertainties, and suffering from the disease, 
are all of these risk factors that could put people on the edge of a mental 
breakdown and at worst trigger the suicidal act.  

At the very moment, with a shortage of health resources and piling up 
pressure on people's mental wellness, such an automatic suicidal risk
detection system can benefit both ends.The system is also not
limited to predicting social media posts, with different sources of data fed 
into the system, it can also be applicable in other circumstances such as 
clinical text records.
""")

# Model Performance
st.header("Model Performance")
st.image(Image.open(MODELFULL_PERF_FIG), use_column_width=True)

# Methodology
st.header("Pipeline")
st.markdown("""Our model is trained on 230k social media posts from reddit. The project pipeline is shown as below.""")
st.image(Image.open(STUDY_DSG), use_column_width=True)

