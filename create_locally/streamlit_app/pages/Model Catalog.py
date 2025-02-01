import streamlit as st
import pandas as pd
import json
from datetime import datetime
from snowflake.ml.registry import registry

# Create model registry
model_registry = st.session_state.model_registry

st.title("Model Catalog")
st.write(f"This page allows the user to view various aspects of the models saved in the model registry.\
         Select a model to see the available versions.")


# Select model
if model_name := st.selectbox("Model", model_registry.show_models().name):
    st.write(model_registry.get_model(model_name).show_versions())
    
    st.subheader('Select a version to view more details')
    version = st.selectbox("Version", model_registry.get_model(model_name).show_versions().name)



version = f'{version}'
print(version)
# Create 2 columns
col1, col2, col3 = st.columns(3)

# Get DF of available models
models_df = model_registry.get_model('INSURANCE_CHARGES_PREDICTION').show_versions()

######## Write Model stats
col1.subheader("Stats")

# Error Metric
try:
    col1.write("**MAPE:** " + str(round(model_registry.get_model(model_name).version((f'"{version}"')).get_metric('mean_abs_pct_err'),4)))
except Exception as e:
    col1.write(f'**MAPE:** Not available{e}')

try:
    model_date = str(models_df.loc[models_df['name'] == version, 'created_on'].iloc[0])
    date_object = datetime.strptime(model_date, "%Y-%m-%d %H:%M:%S.%f%z")
except:
    col1.write('**Created on:** Not available')
                                
# Date created
col1.write(f'**Created on:** {date_object.strftime("%B %d, %Y %H:%M:%S")}')



# Write the expected inputs

col2.subheader("Expected Inputs")

try:
    # Access the user_data column of the dataframe and turn into a json object
    user_data_json = json.loads(models_df.loc[models_df['name'] == version, 'user_data'].iloc[0])
    # Input features
    EXPECTED_MODEL_INPUTS = []

    # Access the function where name == 'predict'
    predict_function = next((func for func in user_data_json['snowpark_ml_data']['functions'] if func['name'] == 'PREDICT'), None)
    if predict_function:
        # Loop through the expected inputs of the model and add them to the list
        for input in predict_function['signature']['inputs']:
            EXPECTED_MODEL_INPUTS.append(input['name'])


    # Show expected inputs
    expected_string = ''
    for input in EXPECTED_MODEL_INPUTS:
        expected_string = expected_string + (f"\n- {input}\n")

    col2.markdown(f"""
                <div style="overflow-y: scroll; height: 300px;">
                {expected_string}
                """, unsafe_allow_html=True)

except:
    col2.write('Expected inputs not available for this model')

col3.subheader("Expected Outputs")
try:
    EXPECTED_MODEL_OUTPUTS = []
    # Access the function where name == 'predict'
    predict_function = next((func for func in user_data_json['snowpark_ml_data']['functions'] if func['name'] == 'PREDICT'), None)
    if predict_function:
        # Loop through the expected inputs of the model and add them to the list
        for input in predict_function['signature']['outputs']:
            EXPECTED_MODEL_OUTPUTS.append(input['name'])

    
    # Show expected outputs
    expected_output_string = ''
    for input in EXPECTED_MODEL_OUTPUTS:
        expected_output_string = expected_output_string + (f"\n- {input}\n")

    col3.markdown(f"""
                <div style="overflow-y: scroll; height: 300px;">
                {expected_output_string}
                """, unsafe_allow_html=True)
    
except:
    col3.write('Expected outputs not available for this model')

