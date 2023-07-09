import streamlit as st
import requests
from PIL import Image

# Add some information about the service
st.title("Loan Approval Prediction")
st.subheader("Just enter variabel below then click Predict button :sunglasses:")

# Create form of input
with st.form(key = "Loan_form"):
    # Create select box input
    # Create box for number input
    Gender = st.selectbox(
        label = "\tGender :",
        options = (
            "Male",
            "Female",
            "Unknown",
        )
    )
    
    Married = st.selectbox(
        label = "\tMarried :",
        options = (
            "No",
            "Yes",
            "Unknown",
        )
    )
    
    Dependents = st.selectbox(
        label = "\tDependents :",
        options = (
            "0",
            "1",
            "2",
            "3+",
            "Unknown",
        )
    )

    Education = st.selectbox(
        label = "\tEducation :",
        options = (
            "Graduate",
            "Not Graduate",
        )
    )
    
    Self_Employed = st.selectbox(
        label = "\tSelf Employed?",
        options = (
            "No",
            "Yes",
            "Unknown",
        )
    )
    
    ApplicantIncome = st.number_input(
        label = "\tApplicant Income :",
        min_value = 0,
        max_value = 81000,
        help = "Value range from 0 to 81000"
    )
    
    CoapplicantIncome = st.number_input(
        label = "\tCo-applicant Income :",
        min_value = 0,
        max_value = 41667,
        help = "Value range from 0 to 41667"
    )
    
    LoanAmount = st.number_input(
        label = "\tLoan Amount :",
        min_value = 9,
        max_value = 700,
        help = "Value range from 9 to 700"
    )
    
    Loan_Amount_Term = st.number_input(
        label = "\tLoan Amount Term :",
        min_value = 12,
        max_value = 480
    )
    
    Credit_History = st.number_input(
        label = "\tCredit History :",
        min_value = 0,
        max_value = 1
    )
    
    Property_Area = st.selectbox(
        label = "\tProperty Area",
        options = (
            "Urban",
            "Rural",
            "Semiurban",
        )
    )
    
    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "Gender" : Gender,
            "Married" : Married,
            "Dependents" : Dependents,
            "Education" : Education,
            "Self_Employed" : Self_Employed,
            "ApplicantIncome" : ApplicantIncome,
            "CoapplicantIncome" : CoapplicantIncome,
            "LoanAmount" : LoanAmount,
            "Loan_Amount_Term" : Loan_Amount_Term,
            "Credit_History" : Credit_History,
            "Property_Area" : Property_Area
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post(f"http://127.0.0.1:8000/predict", json= form_data).json()

        # parse the prediction result
        if res['status'] == 200:
            st.success(f"Loan Approval Prediction is: {res['prediction']}")
        else:
            st.error(f"Error predicting the data... Please Check Your Code {res}")