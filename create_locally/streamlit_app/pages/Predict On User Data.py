import streamlit as st
import pandas as pd
from snowflake.ml.registry import registry

# Get snowflake session from streamlit session state
session = st.session_state.session


def predict(data):
            prediction = model_version.run(data, function_name = 'predict')
            return prediction['PREDICTED_CHARGES']

# Select model
model_registry = st.session_state.model_registry
model_version = model_registry.get_model('INSURANCE_CHARGES_PREDICTION').default



# Collect user inputs: age, gender, bmi, children, smoker, region, medical_history, family_medical_history,
# exercise_frequency, occupation, coverage level

st.subheader('User Input Form')
with st.form("user_input_form"):
    st.write("Select a value for each dimension to see what the model would predict")
    st.write("When submitted, these values run through the preprocessing and prediction pipeline \
             that is saved in Snowflake's Model Registy ")

    # Age
    age = st.slider("Age", 0, 100, 40)

    # Gender
    gender = st.selectbox('Gender', ['Male','Female'])

    # BMI
    bmi = st.slider("BMI", 10, 55, 34)

    # Children
    children = st.slider("Children", 0, 10, 2)

    # Smoker
    smoker = st.selectbox('Smoker', options=['No', 'Yes'])

    # Region
    options = ['Northwest', 'Northeast', 'Southwest', 'Southeast']
    region = st.selectbox('Region', options)

    # Medical history
    medical_options= ['None','Heart Disease','Diabetes','High Blood Pressure']
    medical_history = st.selectbox("Medical History", medical_options).replace(" ", "_")

    # Family medical history
    family_medical_history = st.selectbox("Family Medical History", medical_options).replace(" ", "_")


    # Exercise frequency
    exercise_frequency = st.selectbox("Exercise Frequency", ['Never','Rarely','Occasionally','Frequently'])

    # Occupation
    occupation = st.selectbox("Occupation", ['Blue Collar','White Collar','Student', 'Unemployed']).replace(" ", "_")

    # Coverage level
    coverage_level = st.selectbox("Coverage Level", ['Basic','Standard','Premium'])
    
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")



    if submitted:
        # create a dataframe with the user inputs
        user_input = pd.DataFrame({
            'AGE': [age],
            'GENDER': [gender.upper()],
            'BMI': [bmi],
            'CHILDREN': [children],
            'SMOKER': [smoker.upper()],
            'REGION': [region.upper()],
            'MEDICAL_HISTORY': [medical_history.upper()],
            'FAMILY_MEDICAL_HISTORY': [family_medical_history.upper()],
            'EXERCISE_FREQUENCY': [exercise_frequency.upper()],
            'OCCUPATION': [occupation.upper()],
            'COVERAGE_LEVEL': [coverage_level.upper()]
        })

        st.write(user_input)
        # put in a spinner while the predict is happening
        with st.spinner('Predicting...'):
            st.write(predict(user_input))
