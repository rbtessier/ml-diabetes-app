import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
#import shap

def display_feature_importance(model):
        # Get feature importances (absolute values of coefficients)
        importances = np.abs(model.coef_[0])

        # Associate importances with feature names
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

        # Sort features by importance in descending order
        feature_importance = feature_importance.sort_values('Importance', ascending=False)

        # Display the feature importances
        st.dataframe(feature_importance)


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

# Compute SHAP values
#explainer = shap.LinearExplainer(model, X_train)
#shap_values = explainer.shap_values(X)

st.title('Diabetes Prediction App')

st.write("""
This app predicts whether a patient has diabetes based on specific health factors. The features used for the prediction are:

- Pregnancies: Number of times pregnant.
- PlasmaGlucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
- DiastolicBloodPressure: Diastolic blood pressure (mm Hg).
- TricepsThickness: Triceps skin fold thickness (mm).
- SerumInsulin: 2-Hour serum insulin (mu U/ml).
- BMI: Body mass index (weight in kg/(height in m)^2).
- DiabetesPedigree: Diabetes pedigree function.
- Age: Age (years).

After the prediction, you can ask the model to explain its prediction by clicking the 'Explain yourself!' button. The model will show which features contributed more to the prediction (local feature importance) and how each feature typically affects predictions (global feature importance).
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
    prediction_proba = model.predict_proba(input_data)

    

    if prediction[0] == 0:
        st.write('The patient is not predicted to have diabetes.')
    else:
        st.write('The patient is predicted to have diabetes.')

    st.write('The predicted probability is ', prediction_proba)

    display_feature_importance(model)


    #if st.button("Explain Yourself!"):
        # Calculate and display feature importance
    #    display_feature_importance(model)

    #if st.button('Explain yourself!'):
    #    # Display local SHAP values
    #    shap_force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1], input_data, matplotlib = False)#

    #    st.plotly_chart(shap_force_plot)

        # Display global feature importance
#        shap.summary_plot(shap_values, features, plot_type="bar")

    #    # Display global feature importance
    #    fig, ax = plt.subplots()
    #    shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
    #    st.pyplot(fig)
