import streamlit as st
import pickle
import numpy as np
import pandas as pd
from keras.models import model_from_json

# =========================
# Load model + scaler
# =========================
with open("titanic_ann_model.pkl", "rb") as f:
    model_data = pickle.load(f)

# Reconstruct model
model = model_from_json(model_data['architecture'])
model.set_weights(model_data['weights'])
scaler = model_data['scaler']

# =========================
# Streamlit App
# =========================
st.title("Titanic Survival Prediction")

st.markdown("Fill in the details below to predict survival:")

# User inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=32.0)
sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Map embarkation to one-hot
embarked_1 = 1 if embarked == "Q" else 0
embarked_2 = 1 if embarked == "S" else 0

# Prepare input array in correct order
X_input = np.array([[pclass, age, sibsp, parch, fare, sex, embarked_1, embarked_2]])

# Scale input
X_scaled = scaler.transform(X_input)

# Predict
prediction_prob = model.predict(X_scaled)[0][0]
prediction = 1 if prediction_prob > 0.5 else 0

# Display result
st.subheader("Prediction Result")
st.write(f"Predicted Survival Probability: {prediction_prob:.2f}")
st.write("Predicted Class:", "Survived" if prediction == 1 else "Did not survive")
