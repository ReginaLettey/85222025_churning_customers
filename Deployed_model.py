#Importing libraries
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import sklearn as sk
import streamlit as st


# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Sample feature names
feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

# Streamlit app
st.title("Churn Prediction App")

# Input form for user to enter feature values
user_input = {}
for feature in feature_names:
    user_input[feature] = st.text_input(f"Enter {feature}", "")

# Convert user input to DataFrame
user_df = pd.DataFrame([user_input])

# Preprocess user input
user_input_processed = preprocess_input(user_df)

# Make predictions
prediction = model.predict(user_input_processed)

# Display prediction
st.subheader("Prediction")
st.write("Churn: ", prediction[0])






    
    