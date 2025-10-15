import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load the trained model
# -----------------------------
model = joblib.load("best_model.pkl")  # Replace with your model file name

st.title("🩺 Breast Cancer Classification App")
st.write("Enter the feature values below to predict the tumor class (0, 1, 2, or 3).")

# -----------------------------
# Feature input fields
# -----------------------------
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst'
]

# -----------------------------
# Create input boxes in columns
# -----------------------------
input_data = []
for i, feature in enumerate(feature_names):
    value = st.number_input(f"{feature}", min_value=0.0, step=0.01, format="%.4f")
    input_data.append(value)

# -----------------------------
# Predict button
# -----------------------------
if st.button("🔍 Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    st.subheader("🎯 Predicted Class:")
    st.success(f"The model predicts class: {prediction}")

    st.info("Class Meaning:")
    st.write("""
    - **0** → Type 0  
    - **1** → Type 1  
    - **2** → Type 2  
    - **3** → Type 3  
    """)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed by Pratik Banarse | Breast Cancer Classifier")
