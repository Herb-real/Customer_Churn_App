# Import Key libraries and packages
import gradio as gr

import pandas as pd
import numpy as np

import os
import pickle

import xgboost as xgb
from xgboost import XGBClassifier

# Create key lists
churn_inputs = ["TotalCharges", "MonthlyCharges", "tenure", "StreamingTV", "PaperlessBilling", "DeviceProtection", "TechSupport", "InternetService", "OnlineSecurity", "StreamingMovies", "PaymentMethod", "Dependents", "Partner",
                "tenure_group", "OnlineBackup", "gender", "SeniorCitizen", "MultipleLines", "Contract", "PhoneService"]



# Useful functions
# Load ML toolkit
def load_toolkit(file_path = "C:/Users/LPM/Desktop/Customer_Churn_App/src/app.py"):
    """"_summary_
    
    Args:
        file_path (_type_, optional): _description_. Defaults to'src'.
        
    Returns:
        returns machine learning items
    """
    with open(file_path, "rb") as file:
        loaded_toolkit = pickle.load(file)
        return loaded_toolkit
    
# Import the toolkit
loaded_toolkit = load_toolkit()

# import the scaler
scaler = loaded_toolkit["scaler"]

# Import the model
model = XGBClassifier()
model.load_model()

# Function to process Input and return prediction
def churn_predict(*args, scaler = scaler, model = model):
    """"_summary_
    
    Args:
        scaler (_type_, optional): _description_. Defaults to scaler.
        model (_type_, optional): _description_. Defaults to model
    """
    # Convert inputs into a dataframe
    input_data = pd.DataFrame([args], columns=churn_inputs)
    
    #Scale the numeric columns
    input_data[churn_inputs] = scaler.transform(input_data[churn_inputs])
    
    # Make the prediction
    model_output = model.predict_proba(input_data)


# App Interface
# Inputs
TotalCharges = gr.Number(...)
MonthlyCharges = gr.Number(...)
tenure = gr.Number(...)
StreamingTV = gr.Dropdown(["Select StreamingTV", "No", "Yes", "No Internet Service"], info="Select StreamingTV")
PaperlessBilling = gr.Dropdown(["Select PaperlessBilling", "No", "Yes"])
DeviceProtection = gr.Dropdown(["Select Device Protection", "No", "Yes"])
TechSupport = gr.Dropdown(["Select TechSupport", "No", "Yes"])
InternetService = gr.Dropdown(["Select Internet Service", "DSL", "Fiber optic", "No"])
OnlineSecurity = gr.Dropdown(["Select OnlineSecurity", "No", "Yes", "No internet service"])
StreamingMovies = gr.Dropdown(["Select StreamingMovies", "No", "Yes", "No internet service"])
PaymentMethod = gr.Dropdown(["Select PaymentMethod", "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
Dependents = gr.Dropdown(["Select Dependents", "No", "Yes"])
Partner = gr.Dropdown(["Select Partner", "Yes", "No"])
tenure_group = gr.Dropdown(["Select tenure_group", "less_than_1yr", "less_than_3yr", "less_than_4yr", "less_than_2yr", "greater_than_5yr", "less_than_5yr"])
OnlineBackup = gr.Dropdown(["Select OnlineBackup", "Yes", "No", "No internet service"])
gender = gr.Dropdown(["Select gender", "Female", "Male"])
SeniorCitizen = gr.Dropdown(["Select SeniorCitizen", "No", "Yes"])
MultipleLines = gr.Dropdown(["Select MultipleLines", "No phone service", "No", "Yes"])
Contract = gr.Dropdown(["Select Contract", "Month-to-month", "One year", "Two year"])
PhoneService = gr.Dropdown(["Select PhoneService", "No", "Yes"])


# Output
gr.Interface(inputs = [TotalCharges, MonthlyCharges, tenure, StreamingTV, PaperlessBilling, DeviceProtection, TechSupport, InternetService, OnlineSecurity, StreamingMovies, PaymentMethod, Dependents, Partner,
                         tenure_group, OnlineBackup, gender, SeniorCitizen, MultipleLines, Contract, PhoneService],
             
             outputs= gr.Label("Awaiting Submission..."), 
             fn= churn_predict, title = "App For Customer Churn Prediction").launch(inbrowser=True, show_error=True, share=True)
