# Import python packages
import streamlit as st
from snowflake.snowpark import Session
from snowflake.ml.registry import registry
import json
from datetime import datetime
import pandas as pd

# Connect to Snowflake
@st.cache_resource()
def connect_to_snowflake():
    connection_parameters = json.load(open('../connection.json'))
    session = Session.builder.configs(connection_parameters).create()
    session.use_database('INSURANCE')
    session.use_schema('ML_PIPE')
    session.use_warehouse('COMPUTE_WH')
    print("Connected to Snowflake successfully")
    if 'session' not in st.session_state:
        st.session_state.session = session
    return session


session = connect_to_snowflake()

# Create model registry and add to session state
model_registry = registry.Registry(session=session, database_name=session.get_current_database(), schema_name=session.get_current_schema())

if 'model_registry' not in st.session_state:
    st.session_state.model_registry = model_registry

if 'session' not in st.session_state:
    st.session_state.session = session

# Small intro
st.title("Insurance ML Pipeline")
st.write(f"This Streamlit app allows the user to view various aspects of the ML pipeline built for the insurance dataset.")


gold_df = session.table('INSURANCE_GOLD').limit(600).to_pandas()

# Create a scatterplot with 'PREDICTED_CHARGES' on the x-axis and 'CHARGES' on the y-axis
st.subheader('Scatterplot of Predicted vs Actual Charges')
st.scatter_chart(gold_df, x='PREDICTED_CHARGES', y='CHARGES')
