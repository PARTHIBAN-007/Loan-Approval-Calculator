# Objectives : 
The Loan Price Calculator aims to determine a user's eligibility for a loan based on their requested amount and a predefined maximum eligible amount. It helps users understand their borrowing capacity and provides clarity on whether they qualify for the loan they seek.

## Key Features:
1. Determine Eligibility: Assess if the expected loan amount is within the eligible range.
2. Provide Clear Feedback: Inform users whether they are eligible or not for the loan based on their input.
3. Promote Financial Awareness: Educate users about their borrowing limits and responsible loan practices.
4. User-Friendly Experience: Ensure ease of use through simple inputs and straightforward output messages.

This calculator serves as a useful tool for individuals seeking loans, helping them make informed financial decisions.


## Images of Loan Price Calculator
<img src ="./Assets/Loan price Calculator.png">
<img src ="./Assets/Approval.png">
<img src ="./Assets/Not Eligible.png">


## Repository Structure


1. **data**: Contains the Data

2. **model**: Directory for saving and loading the model.pkl file.

3. **Experiments**: Python notebooks for cleaning, preprocessing, feature engineering ,Model Building , Training and evaluattion 
       

5. **src**: Main source code directory with the following subfolders:
    - **DatPprocessing**: Functionality to preprocess, feature engineering, modeling on a raw dataset
    - **BuildModel**: Creates and saves the model.pkl file from the preprocessed dataset
    - **predict**: Predictions of the saved model.pkl on new user input

6. **app.py**: Streamlit frontend

7. **Dockerfile**: Configuration for setting up the project in a Docker container.


# Setting up the Project

**With Docker**
1. Clone the repository
```
git clone https://github.com/PARTHIBAN-007/Uber-ETA.git
```
2. build the Docker Image
```
docker build -t ml-app .
```
3. Run the following command to use the app
```
docker run ml-app
```

**Without Docker**
1. Clone the repository
```
git clone https://github.com/PARTHIBAN-007/Uber-ETA.git
```
2. install all the libraries and dependencies
```
pip install -r requirements.txt
```
3. Run the following command to use the app
```
streamlit run main.py
```

