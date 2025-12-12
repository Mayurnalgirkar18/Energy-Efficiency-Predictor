# ⚡ Energy Efficiency Prediction App (Supervised Learning)

This project predicts "Heating Load" and "Cooling Load" of buildings based on 8 architectural and design-related features.  
It uses Machine Learning (Random Forest Regression) and includes a fully working "Streamlit web app" for real-time predictions.

---

# Project Features

-  *Machine Learning Model* (Random Forest Regressor)  
-  *Predicts two outputs:*
  - Heating Load  
  - Cooling Load  
-  *Interactive Streamlit Web Application* 
-  *Visualizations* for model evaluation  
-  Clean project structure  
-  Model saved using `pickle` for deployment  
-  No label encoding required (all features are numeric)

---

# 📂 Dataset Information

The dataset contains building energy parameters:

| Feature                       | Description |
|------------------------------|-------------|
| relative_compactness         | Compactness of the building |
| surface_area                 | Total external surface area |
| wall_area                    | External wall area |
| roof_area                    | Roof area |
| overall_height               | Height of the building |
| orientation                  | Building orientation (2–5) |
| glazing_area                 | Glazing area ratio |
| glazing_area_distribution    | Distribution type (0–5) |
| heating_load (target)        | Energy required for heating |
| cooling_load (target)        | Energy required for cooling |

# Machine Learning Workflow

1. Load dataset  
2. Preprocess data  
   - Split into train & test  
   - Scale features using `StandardScaler`  
3. Train model using **RandomForestRegressor**  
4. Evaluate model  
   - MSE  
   - R² Score  
5. Save model & scaler (`model.pkl`, `scaler.pkl`)  
6. Deploy using Streamlit

Stremlit Web App

The app allows users to input building parameters and get real-time predictions.

 Run the Streamlit App

python -m streamlit run app.py
