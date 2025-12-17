import streamlit as st
import pickle
import numpy as np
from keras.models import model_from_json

# =========================
# Load model + scaler (cached for faster reruns)
# =========================
@st.cache_resource
def load_model():
    with open("titanic_ann_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    model = model_from_json(model_data['architecture'])
    model.set_weights(model_data['weights'])
    scaler = model_data['scaler']
    return model, scaler

model, scaler = load_model()

# =========================
# Streamlit App UI
# =========================
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

st.title("🚢 Titanic Survival Predictor")
st.markdown("""
Predict whether a passenger would survive the Titanic disaster.  
Fill out the passenger details below.
""")

# =========================
# Input Form
# =========================
with st.form("passenger_form"):
    st.subheader("Passenger Details")

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = First Class, 2 = Second Class, 3 = Third Class")
        age = st.slider("Age", 0, 100, 30, help="Passenger age in years")
        sibsp = st.slider("Siblings / Spouses aboard", 0, 10, 0)
        parch = st.slider("Parents / Children aboard", 0, 10, 0)

    with col2:
        fare = st.number_input("Fare ($)", min_value=0.0, max_value=1000.0, value=32.0, step=1.0)
        sex = st.radio("Sex", ["Female", "Male"])
        embarked = st.selectbox("Port of Embarkation", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])

    submitted = st.form_submit_button("Predict Survival")

# =========================
# Map inputs to model format
# =========================
if submitted:
    sex_num = 1 if sex == "Male" else 0
    embarked_1 = 1 if embarked.startswith("Q") else 0
    embarked_2 = 1 if embarked.startswith("S") else 0

    X_input = np.array([[pclass, age, sibsp, parch, fare, sex_num, embarked_1, embarked_2]])
    X_scaled = scaler.transform(X_input)
    prediction_prob = model.predict(X_scaled)[0][0]
    prediction = "Survived" if prediction_prob > 0.5 else "Did not survive"

    # =========================
    # Display results
    # =========================
    st.subheader("Prediction Result")
    st.markdown(f"""
    **Predicted Survival:**  
    <span style='color:{"green" if prediction=="Survived" else "red"}; font-size:22px;'>{prediction}</span>

    **Survival Probability:** {prediction_prob:.2f}
    """, unsafe_allow_html=True)

    # Add additional feedback
    if prediction == "Survived":
        st.balloons()
    else:
        st.warning("Sadly, this passenger likely did not survive.")
