# --- Deploy with Streamlit ---
import tensorflow as tf
import numpy as np
import pandas as pd

# streamlit run app.py
import streamlit as st

# load the trained model and medians
model = tf.keras.models.load_model('diabetes_model.keras')
medians = pd.read_pickle('train_medians.pkl')


# Page configuration
st.set_page_config(layout="wide")

# Feature selection on sidebar
def get_user_input():
    pregnancy = st.sidebar.number_input('Pregnancies (0 if none)', min_value=0, max_value=17, step=1, value=0)
    glucose = st.sidebar.number_input('Glucose level (0-199)', min_value=0.0, max_value=199.0, step=1.0, value=0.0)
    blood = st.sidebar.number_input('Blood Pressure (mm Hg)',  min_value=0.0, max_value=122.0, step=1.0, value=0.0)
    skin = st.sidebar.number_input('Skin Thickness (mm)',  min_value=0.0, max_value=99.0, step=1.0, value=0.0)
    insulin = st.sidebar.number_input('Insulin (mu U/ml)',  min_value=0.0, max_value=846.0, step=1.0, value=0.0)
    bmi = st.sidebar.number_input('BMI (kg/m^2)',  min_value=0.0, max_value=67.1, step=1.0, value=0.0)
    pedigree = st.sidebar.number_input('Diabetes Pedigree Function',  min_value=0.00, max_value=2.42, step=0.100, value=0.00, format="%.4f")
    age = st.sidebar.number_input('Age (21-81 years)',  min_value=21, max_value=81, step=1, value=21)

    user_data = {
        'Feature1': pregnancy,
        'Feature2': glucose,
        'Feature3': blood, 
        'Feature4': skin,
        'Feature5': insulin,
        'Feature6': bmi,
        'Feature7': pedigree,
        'Feature8': age
    }
    
    return user_data

# Centered title
st.markdown("<h1 style='text-align: center;'>Diabetes Risk Prediction</h1>", unsafe_allow_html=True)
    
# Prediction Interface
st.header("Predict Risk")

# User inputs from sidebar
user_data = pd.DataFrame([get_user_input()])

# Predict button
if st.button("Predict"):
    # Fix the missing data
    col_to_fix = [f'Feature{i}' for i in range (2, 7)]
    user_data[col_to_fix] = user_data[col_to_fix].replace(0, np.nan)
    user_data = user_data.fillna(medians)

    # Predict
    prediction = model.predict(user_data.values.astype(np.float32))
    prob = float(prediction[0][0])

    # Display results
    st.subheader("Risk: ")
    if prob > 0.5:
        st.error(f"Prediction: Diabetic ({prob:.2%})")
    else:
        st.success(f"Prediction: Not Diabetic ({prob:.2%})")