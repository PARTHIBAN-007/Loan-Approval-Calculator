import pickle
import pandas as pd

def predict(x):
    with open("./Model/model1.pkl", "rb") as f:
        load_artifacts = pickle.load(f)

    model = load_artifacts['model']
    scaler = load_artifacts['scaler']
    label_encoders = load_artifacts['label'] 

    # Encode categorical columns using label encoders
    for column, le in label_encoders.items():
        if column in x.columns:
            x[column] = le.transform(x[column])
        else:
            raise ValueError(f"Column {column} not found in input DataFrame.")
    numeric_columns = ['person_age', 'person_income', 'person_emp_length', 'loan_int_rate',
        'loan_percent_income', 'cb_person_cred_hist_length']

    # Scale the features
    x[numeric_columns] = scaler.transform(x[numeric_columns]) 

    # Make predictions
    predictions = model.predict(x)
    return predictions

# Creating the DataFrame with all relevant columns
df = pd.DataFrame({
    "person_age": [22],
    "person_income": [33000],
    "person_home_ownership": ["RENT"],  # Categorical
    "person_emp_length": [6.0],
    "loan_intent": ["PERSONAL"],         # Categorical
    "loan_grade": ["B"],                  
    "loan_int_rate": [11.12],
    "loan_percent_income": [0.3],
    "cb_person_default_on_file": ["N"],   # Categorical
    "cb_person_cred_hist_length": [2]
})


# Make predictions
predictions = predict(df)
print("Predictions:", predictions)
