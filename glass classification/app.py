import streamlit as st
import numpy as np
import pandas as pd
import joblib 

# loading the saved components
model = joblib.load('gradient_boosting_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Glass Classification Model")
st.write("Creating a model that classifies glass")

# loading features

RI = st.slider('Reractive Index: ',1.5112,1.5339)
Na=  st.slider('Sodium: ',10.7300,17.3800)
Mg = st.slider('Magnesium: ',0.0000,4.4900)
Al = st.slider('Aluminium: ',0.2900,3.5000)
Si = st.slider('Silicon: ',68.8100,75.4100)
K = st.slider('Potassium: ',0.0000,6.2100)
Ca = st.slider('Calcium: ',5.4300,16.1900)
Ba = st.slider('Barium: ',0.0000,3.1500)
Fe = st.slider('Iron: ',0.0000,0.5100)


# preparing input feature for model
features = np.array([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])
scaled_features=scaler.transform(features)

#prediction

if st.button('Predict Glass Type'):
    prediction_encoded=model.predict(scaled_features)
    prediction_label=label_encoder.inverse_transform(prediction_encoded)[0]

    st.success('Predicted glass type: {}'.format(prediction_label))
