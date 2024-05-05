# import the required libraries
import numpy as np
import pandas as pd
import pickle
import streamlit as st
# provide header name to browser
st.set_page_config(page_title='Iris Project Pratik', layout='wide')

# Add a title in body of browser
st.title('Iris Project')

# continue in code
# take input from user

sep_len = st.number_input('Sepal Length: ', min_value= 0.00, step=0.01 )
sep_wid = st.number_input('Sepal Width: ', min_value= 0.00, step=0.01 )
pet_len = st.number_input('Petal Length: ', min_value= 0.00, step=0.01 )
pet_wid = st.number_input('Petal Width: ', min_value= 0.00, step=0.01 )


# Add a button for predictions
submit= st.button('Predict')

# Add subheader for predictions
st.subheader('Predictions Are :')

# Create a function to predict species along with probability

def predict_species(scaler_path, model_path):
    with open(scaler_path,'rb') as file1:
        scaler= pickle.load(file1)
    with open(model_path, 'rb') as file2:
        model= pickle.load(file2)
    # Input from web page
    dct = {'SepalLengthCm': [sep_len],
           'SepalWidthCm': [sep_wid],
           'PetalLengthCm': [pet_len],
           'PetalWidthCm': [pet_wid]}
    xnew= pd.DataFrame(dct)
    xnew_pre= scaler.transform(xnew)
    pred= model.predict(xnew_pre)
    probs= model.predict_proba(xnew_pre)
    max_prob= np.max(probs)
    return pred, max_prob

# show the results in streamlit

if submit:
    scaler_path= 'Notebook/scaler.pkl'
    model_path= 'Notebook/model.pkl'
    pred, max_prob= predict_species(scaler_path,model_path)
    st.subheader(f'Predicted Species is: {pred[0]}')
    st.subheader(f'Probability of Prediction is: {max_prob:.4f}')
    st.progress(max_prob)