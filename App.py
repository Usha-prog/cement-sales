import pandas as pd
import streamlit as st
import numpy as np
from autots import AutoTS

st.title('Forecasting the Cement Sales')
uploaded_file = st.file_uploader(" ", type=['xlsx'])

if uploaded_file is not None:     
    cement = pd.read_excel(uploaded_file)
    cement['Month'] = cement['Month'].apply(lambda x: x.strftime('%B-%Y'))
    st.write("Plese wait for the forecasting result... Model is working on it")
    
    mod = AutoTS(forecast_length=12, frequency='M', prediction_interval = 0.90,
             ensemble= None, model_list = 'univariate', max_generations = 3, num_validations= 2,
             no_negatives = True,n_jobs = 'auto')

    mod = mod.fit(cement, date_col='Month', value_col='Sales')
    prediction = mod.predict()
    
    forecast = prediction.forecast
    
    st.subheader("From AutoTS model")
   
    st.write("Forecast of Sales for next 12 months: ", forecast)
