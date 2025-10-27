import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Load model and scaler
rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))
scaler = pickle.load(open('scaler_save.pkl', 'rb'))
print("Random Forest Model Saved Succesfully")

# User Input

def user_input_features():
    relative_compactness = st.number_input("Relative Compactness", min_value=0.5, max_value=1.0, value=0.75, step=0.01)
    surface_area = st.number_input("Surface Area", min_value=400.0, max_value=900.0, value=600.0, step=1.0)
    wall_area = st.number_input("Wall Area", min_value=200.0, max_value=400.0, value=300.0, step=1.0)
    roof_area = st.number_input("Roof Area", min_value=50.0, max_value=350.0, value=150.0, step=1.0)
    overall_height = st.number_input("Overall Height", min_value=3.0, max_value=7.0, value=3.5, step=0.1)
    orientation = st.selectbox("Orientation", options=[2,3,4,5])
    glazing_area = st.number_input("Glazing Area", min_value=0.0, max_value=0.4, value=0.1, step=0.01)
    glazing_area_distribution = st.selectbox("Glazing Area Distribution", options=[0,1,2,3,4])

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
    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

# -------------------------------
# 3. Scale input
# -------------------------------
input_scaled = scaler.transform(input_df)

# -------------------------------
# 4. Prediction
# -------------------------------
if st.button("Predict Heating Load"):
    prediction = rf_model.predict(input_scaled)
    st.success(f"Predicted Heating Load: {prediction[0]:.3f}")
