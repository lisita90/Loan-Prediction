import streamlit as st
import pandas as pd
import pickle

# Load the model and label encoders
with open('knn_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Function to preprocess the input data
def preprocess_input(data):
    # Convert categorical data to numeric using label encoders
    for column, le in label_encoders.items():
        if column != 'Loan_Status':  # Exclude the target column
            data[column] = le.transform(data[column])
    return data

# Function to predict loan status
def predict_loan_status(data):
    # Preprocess the input data
    data = preprocess_input(data)
    # Predict using the loaded model
    prediction = loaded_model.predict(data)
    return prediction

# Define the main function for the Streamlit app
def main():
    st.title('Loan Prediction App')

    # Collect user input using Streamlit components
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Married', ['Yes', 'No'])
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    applicant_income = st.number_input('Applicant Income')
    coapplicant_income = st.number_input('Coapplicant Income')
    loan_amount = st.number_input('Loan Amount')
    loan_amount_term = st.number_input('Loan Amount Term')
    credit_history = st.selectbox('Credit History', [0, 1])
    property_area = st.selectbox('Property Area', ['Rural', 'Semiurban', 'Urban'])
    dependents = st.number_input('Dependents', min_value=0, max_value=10)

    # Create a dictionary from the user inputs
    input_data = {
        'Gender': [gender],
        'Married': [married],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area],
        'Dependents': [dependents]
    }

    # Convert the dictionary into a DataFrame
    input_df = pd.DataFrame(input_data)

    # Make predictions
    if st.button('Predict'):
        prediction = predict_loan_status(input_df)
        st.write('Prediction:', prediction[0])

# Run the main function
if __name__ == '__main__':
    main()
