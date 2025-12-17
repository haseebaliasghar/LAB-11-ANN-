# =========================================================
# TITANIC SURVIVAL PREDICTOR - STREAMLIT APP
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =============================
# Load trained model and scaler
# =============================
with open("titanic_ann_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# These are the columns used during training
training_columns = [
    'Pclass','Age','sibsp','Parch','Fare',
    'Sex_female','Embarked_Q','Embarked_S'
]

# =============================
# Streamlit UI
# =============================
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict the probability of survival.")

# --- User inputs ---
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1,2,3])
sex = st.selectbox("Sex", ["male","female"])
age = st.number_input("Age", min_value=0, max_value=120, value=18)
sibsp = st.number_input("Number of siblings/spouses aboard", min_value=0, value=0)
parch = st.number_input("Number of parents/children aboard", min_value=0, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.05, step=0.01)
embarked = st.selectbox("Port of Embarkation", ["C","Q","S"])

# --- Prepare input dataframe ---
input_df = pd.DataFrame({
    'Pclass':[pclass],
    'Age':[age],
    'sibsp':[sibsp],
    'Parch':[parch],
    'Fare':[fare],
    'Sex':[sex],
    'Embarked':[embarked]
})

# One-hot encode categorical features
input_df = pd.get_dummies(input_df, columns=['Sex','Embarked'], drop_first=False)

# Add any missing columns (fill with 0)
for col in training_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns exactly like training
input_df = input_df[training_columns]

# Scale numeric columns
numeric_cols = ["Pclass","Age","sibsp","Parch","Fare"]
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# =============================
# Prediction
# =============================
if st.button("Predict Survival Probability"):
    pred_prob = model.predict(input_df)[0][0]
    pred_class = int(pred_prob > 0.5)
    
    st.write(f"**Predicted probability of survival:** {pred_prob:.2f}")
    st.write("**Prediction:**", "✅ Survived" if pred_class==1 else "❌ Did not survive")
