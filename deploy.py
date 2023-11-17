# Importing libraries
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load the trained model and scaler
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Features
feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

# Streamlit app
st.title("Churn Prediction App")
st.write("Enter customer attributes to predict churn")

# Define input fields for user input
gender = st.selectbox("Gender", ['Male', 'Female'])
SeniorCitizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
Partner = st.selectbox("Partner", ['No', 'Yes'])
Dependents = st.selectbox("Dependents", ['No', 'Yes'])
tenure = st.number_input("Tenure", min_value=0, value=100)
PhoneService = st.selectbox("Phone Service", ['No', 'Yes'])
MultipleLines = st.selectbox("Multiple Lines", ['No', 'Yes'])
InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic'])
OnlineSecurity = st.selectbox("Online Security", ['No', 'Yes'])
OnlineBackup = st.selectbox("Online Backup", ['No', 'Yes'])
DeviceProtection = st.selectbox("Device Protection", ['No', 'Yes'])
TechSupport = st.selectbox("Tech Support", ['No', 'Yes'])
StreamingTV = st.selectbox("Streaming TV", ['No', 'Yes'])
StreamingMovies = st.selectbox("Streaming Movies", ['No', 'Yes'])
Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox("Paperless Billing", ['No', 'Yes'])
PaymentMethod = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0, value=1000)
TotalCharges = st.number_input("Total Charges", min_value=0, value=1000)

# Create a button to trigger the prediction
if st.button("Predict Churn"):
    # Prepare the input data for prediction
    input_data = {
        "gender": [1 if gender == 'Female' else 0],
        "SeniorCitizen": [1 if SeniorCitizen == 'Yes' else 0],
        "Partner": [1 if Partner == 'Yes' else 0],
        "Dependents": [1 if Dependents == 'Yes' else 0],
        "tenure": [tenure],
        "PhoneService": [1 if PhoneService == 'Yes' else 0],
        "MultipleLines": [1 if MultipleLines == 'Yes' else 0],
        "InternetService": [1 if InternetService == 'Fiber optic' else 0],
        "OnlineSecurity": [1 if OnlineSecurity == 'Yes' else 0],
        "OnlineBackup": [1 if OnlineBackup == 'Yes' else 0],
        "DeviceProtection": [1 if DeviceProtection == 'Yes' else 0],
        "TechSupport": [1 if TechSupport == 'Yes' else 0],
        "StreamingTV": [1 if StreamingTV == 'Yes' else 0],
        "StreamingMovies": [1 if StreamingMovies == 'Yes' else 0],
        "Contract": [1 if Contract == 'One year' else 0],
        "PaperlessBilling": [1 if PaperlessBilling == 'Yes' else 0],
        "PaymentMethod": [1 if PaymentMethod == 'Mailed check' else 0],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges]
    }

    input_df = pd.DataFrame(input_data)
    scaled_input_data = scaler.transform(input_df)

    # Make the prediction
    prediction = model.predict(scaled_input_data)

    # Calculate the confidence factor
    confidence_factor = prediction.squeeze()

    # Display the prediction and confidence factor
    st.write(f"Predicted Churn: {int(round(float(confidence_factor)))}")
    st.write(f"Confidence Factor: {confidence_factor:.2f}")

    # Display the prediction
    if confidence_factor > 0.5:
        st.warning('Churn: Yes')
    else:
        st.success('Churn: No')

# Create a reset button to clear the input fields
if st.button("Reset"):
    st.experimental_rerun()

