import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import model_from_json

# 1. Load model + scaler from single PKL
with open("titanic_ann_model.pkl", "rb") as f:
    model_data = pickle.load(f)

# Reconstruct the ANN model
model = model_from_json(model_data["architecture"])
model.set_weights(model_data["weights"])

# Load scaler
scaler = model_data["scaler"]

st.title("Titanic Survival Prediction")

# Example: user input for features (adjust these to your dataset)
pclass = st.number_input("Passenger Class (1,2,3)", min_value=1, max_value=3, value=3)
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)

# Convert categorical to numeric
sex_num = 1 if sex == "male" else 0

# Create input array
X_input = np.array([[pclass, sex_num, age, sibsp, parch, fare]])

# Scale input using the loaded scaler
X_scaled = scaler.transform(X_input)

# Make prediction
prediction = model.predict(X_scaled)
survival_prob = prediction[0][0]

st.write(f"Predicted Survival Probability: {survival_prob:.2f}")
st.write("Prediction: **Survived**" if survival_prob > 0.5 else "Prediction: **Did Not Survive**")
