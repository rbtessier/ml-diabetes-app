import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load the diabetes dataset
# Replace the path with the path where your diabetes.csv file is
data = pd.read_csv("diabetes.csv")

# Split the dataset
features = ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI']
X = data[features]
y = data['Diabetic']

# Train a logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Create Streamlit app
st.title('Diabetes Prediction App')

st.write("""
Enter patient health parameters to predict if the patient has diabetes.
""")

# Input sliders
pregnancies = st.slider('Pregnancies', min_value=0, max_value=10, value=5)
plasma_glucose = st.slider('Plasma Glucose', min_value=0, max_value=200, value=100)
diastolic_bp = st.slider('Diastolic Blood Pressure', min_value=0, max_value=150, value=80)
triceps_thickness = st.slider('Triceps Thickness', min_value=0, max_value=100, value=20)
serum_insulin = st.slider('Serum Insulin', min_value=0, max_value=800, value=100)
bmi = st.slider('BMI', min_value=0.0, max_value=60.0, value=30.0, step=0.1)

# 'Predict' button
if st.button('Predict'):
    input_data = np.array([[pregnancies, plasma_glucose, diastolic_bp, triceps_thickness, serum_insulin, bmi]])
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.write('The patient is not predicted to have diabetes.')
    else:
        st.write('The patient is predicted to have diabetes.')
