import streamlit as st
import pandas as pd
import numpy as np
import pickle
from keras.models import model_from_json

# ===============================
# Load trained model & scaler
# ===============================
with open("titanic_ann_model.pkl", "rb") as f:
    model_data = pickle.load(f)

# Rebuild Keras model
model = model_from_json(model_data["architecture"])
model.set_weights(model_data["weights"])
scaler = model_data["scaler"]

# ===============================
# Streamlit App
# ===============================
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Predictor")

st.markdown("""
Enter the passenger details below to predict the probability of survival.
""")

# Sidebar or main input form
with st.form(key="input_form"):
    Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1,2,3])
    Sex = st.selectbox("Sex", ["male","female"])
    Age = st.number_input("Age", min_value=0, max_value=100, value=30)
    SibSp = st.number_input("Number of siblings/spouses aboard", min_value=0, max_value=10, value=0)
    Parch = st.number_input("Number of parents/children aboard", min_value=0, max_value=10, value=0)
    Fare = st.number_input("Fare", min_value=0.0, value=32.0)
    Embarked = st.selectbox("Port of Embarkation", ["C","Q","S"])

    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    # Create input DataFrame
    input_dict = {
        "Pclass": [Pclass],
        "Age": [Age],
        "sibsp": [SibSp],
        "Parch": [Parch],
        "Fare": [Fare],
        "Sex_female": [1 if Sex=="female" else 0],
        "Embarked_Q": [1 if Embarked=="Q" else 0],
        "Embarked_S": [1 if Embarked=="S" else 0]
    }

    input_df = pd.DataFrame(input_dict)

    # Scale numeric features (Pclass, Age, SibSp, Parch, Fare)
    numeric_cols = ["Pclass","Age","sibsp","Parch","Fare"]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Prediction
    pred_prob = model.predict(input_df)[0][0]
    pred_class = int(pred_prob > 0.5)

    st.subheader("Prediction Result")
    st.write(f"**Survival Probability:** {pred_prob*100:.2f}%")
    st.write("**Predicted Class:**", "Survived" if pred_class==1 else "Did Not Survive")
