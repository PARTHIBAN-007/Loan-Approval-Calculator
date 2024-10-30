import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load the model and other artifacts
def load_model():
    with open("./Model/model1.pkl", "rb") as f:
        return pickle.load(f)

load_artifacts = load_model()
model = load_artifacts['model']
scaler = load_artifacts['scaler']
label_encoders = load_artifacts['label']

def predict(x):
    # Encode categorical columns
    for column, le in label_encoders.items():
        if column in x.columns:
            x[column] = le.transform(x[column])
    
    numeric_columns = ['person_age', 'person_income', 'person_emp_length', 'loan_int_rate',
        'loan_percent_income', 'cb_person_cred_hist_length']

    # Scale the features
    x[numeric_columns] = scaler.transform(x[numeric_columns]) 
    # Make predictions
    predictions = model.predict(x)
    return predictions

# Streamlit app
st.title("Loan Prediction App")
col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Person Age", min_value=18, max_value=100, value=22)
    person_income = st.number_input("Person Income", min_value=1000, value=33000)
    person_home_ownership = st.selectbox("Home Ownership", options=["RENT", "OWN", "MORTGAGE"])
    person_emp_length = st.number_input("Person Employment Length (in years)", min_value=0.0, value=6.0)
    loan_intent = st.selectbox("Loan Intent", options=["PERSONAL", "AUTO", "HOME", "EDUCATION"])

    
with col2:
    loan_grade = st.selectbox("Loan Grade", options=["A", "B", "C", "D", "E", "F", "G"])
    loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, value=11.12)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.3)
    cb_person_default_on_file = st.selectbox("Default on File", options=["Y", "N"])
    cb_person_cred_hist_length = st.number_input("Credit History Length (in years)", min_value=0, value=2)

loan_amount = st.number_input("Expected Loan Amount")


# Prepare the DataFrame
if st.button("Predict"):
    input_data = {
        "person_age": [person_age],
        "person_income": [person_income],
        "person_home_ownership": [person_home_ownership],
        "person_emp_length": [person_emp_length],
        "loan_intent": [loan_intent],
        "loan_grade": [loan_grade],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_default_on_file": [cb_person_default_on_file],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length]
    }
    
    df = pd.DataFrame(input_data)

    # Make predictions
    predictions = predict(df)
    predictions =  int(predictions[0])


    if predictions >=loan_amount :
        st.success('You are Eligible for Loan', icon="✅")
    else:    
        st.warning("You are Not Eligible for Loan")

    
    
    # Display the result
    st.write("Maximum Applicable Loan Amount:", predictions)
