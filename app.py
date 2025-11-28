import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
print("Random Forest Model Loaded Successfully")

st.set_page_config(
    page_title="Energy Efficiency Prediction App",
    page_icon="⚡",
    layout="centered"
)

# App Title
st.title("⚡ Energy Efficiency Prediction App")

# Subtitle
st.write("""
Welcome!  
Enter the building parameters below to predict **Heating Load** and **Cooling Load**  
using a trained Machine Learning model.
""")


# User Input
def user_input_features():
    relative_compactness = st.number_input("Relative Compactness", min_value=0.5, max_value=1.0, value=0.75, step=0.01)
    surface_area = st.number_input("Surface Area", min_value=400.0, max_value=900.0, value=600.0, step=1.0)
    wall_area = st.number_input("Wall Area", min_value=200.0, max_value=400.0, value=300.0, step=1.0)
    roof_area = st.number_input("Roof Area", min_value=50.0, max_value=350.0, value=150.0, step=1.0)
    overall_height = st.number_input("Overall Height", min_value=3.0, max_value=7.0, value=3.5, step=0.1)
    orientation = st.selectbox("Orientation", options=[2, 3, 4, 5])
    glazing_area = st.number_input("Glazing Area", min_value=0.0, max_value=0.4, value=0.1, step=0.01)
    glazing_area_distribution = st.selectbox("Glazing Area Distribution", options=[0, 1, 2, 3, 4])

    data = {
        'relative_compactness': relative_compactness,
        'surface_area': surface_area,
        'wall_area': wall_area,
        'roof_area': roof_area,
        'overall_height': overall_height,
        'orientation': orientation,
        'glazing_area': glazing_area,
        'glazing_area_distribution': glazing_area_distribution
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction button
if st.button("Predict Energy Efficiency"):
    prediction = model.predict(input_scaled)

    heating_load = prediction[0][0]
    cooling_load = prediction[0][1]

    st.success(f"🔥 Predicted Heating Load: {heating_load:.3f}")
    st.success(f"❄ Predicted Cooling Load: {cooling_load:.3f}")

#To run programn run this (python -m streamlit run app.py)  in Terminal and make sure streamlit is installed.
