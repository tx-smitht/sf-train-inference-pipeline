# Imports and create a session
from snowflake.snowpark import Session
import snowflake.snowpark
from snowflake.snowpark.dataframe import col as column
import snowflake.snowpark.functions as F
from snowflake.snowpark.functions import sproc
import snowflake.snowpark.types as T
from snowflake import telemetry

import json
import pandas as pd
import numpy as np
from opentelemetry import trace

# Snowpark ML
from snowflake.ml._internal.utils import identifier
from snowflake.ml.registry import registry

from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.xgboost import XGBRegressor
import snowflake.ml.modeling.preprocessing as snowmlpp
from snowflake.ml.modeling.impute import SimpleImputer
from snowflake.ml.modeling.model_selection import GridSearchCV
from snowflake.ml.modeling.metrics import mean_absolute_percentage_error, mean_squared_error

# Create a session
connection_parameters = json.load(open('./connection.json'))
session = Session.builder.configs(connection_parameters).create()
session.use_database('INSURANCE')
session.use_schema('ML_PIPE')
session.use_warehouse('COMPUTE_WH')

@sproc(
        name='predict_write_to_gold',
        stage_location='@ML_PIPE_STAGE', 
        is_permanent=True, 
        replace=True,
        packages=[
                "snowflake-snowpark-python",
                'snowflake-ml-python', 
                'xgboost',
                'pandas',
                "snowflake-telemetry-python",  # Required for tracing
                "opentelemetry-api" # Required for tracing
                ])
def predict_write_to_gold(session: Session) -> str:

        tracer = trace.get_tracer("insurance_model.predict_write_to_gold")

        with tracer.start_as_current_span("predict_write_to_gold") as main_span:
                try:
                        # Read from the stream
                        with tracer.start_as_current_span("read_from_stream"):
                                try:    
                                        df = session.table('STREAM_ON_LANDING').filter(column('METADATA$ACTION') == 'INSERT')

                                except Exception as e:
                                        return (f'Error with reading from stream: {e}')

                        # Standardize values
                        with tracer.start_as_current_span("standardize_values"):
                                try:
                                        # Define Snowflake categorical types and determine which columns to OHE
                                        categorical_types = [T.StringType]
                                        cols_to_ohe = [col.name for col in df.schema.fields if (type(col.datatype) in categorical_types)]
                                        ohe_cols_output = [col + '_OHE' for col in cols_to_ohe]

                                        def fix_values(columnn):
                                                return F.upper(F.regexp_replace(F.col(columnn), '[^a-zA-Z0-9]+', '_'))
                                        
                                        for col in cols_to_ohe:
                                                df = df.na.fill('NONE', subset=col)
                                                df = df.withColumn(col, fix_values(col))

                                except Exception as e:
                                        return (f'Error standardizing values {e}')

                        # Create model registry object and load the default pipeline
                        with tracer.start_as_current_span("load_model_from_registry"):
                                try:
                                        model_registry = registry.Registry(session=session, database_name=session.get_current_database(), schema_name='ML_PIPE')
                                        model_version = model_registry.get_model('INSURANCE_CHARGES_PREDICTION').default

                                except Exception as e:
                                        return (f'Error with creating model registry object: {e}')


                        # Run the pipeline
                        with tracer.start_as_current_span("run_pipeline"):
                                try:
                                        results = model_version.run(df,function_name = 'predict')

                                except Exception as e:
                                        return (f'Error with running model: {e}')

                        # Load the results into the gold table
                        with tracer.start_as_current_span("write_predictions_to_gold"):
                                try:
                                        count = results.count()

                                        cols_to_update = {col: results[col] for col in session.table('INSURANCE_GOLD').columns if 'METADATA_UPDATED_AT' not in col}
                                        metadata_col_to_update = {'METADATA_UPDATED_AT': F.current_timestamp()}
                                        updates = {**cols_to_update, **metadata_col_to_update}
                                        target = session.table('INSURANCE_GOLD')
                                        merge_results = target.merge(results,target['METADATA$ROW_ID'] == results['METADATA$ROW_ID'], \
                                                [F.when_matched().update(updates), F.when_not_matched().insert(updates)])
                                        

                                        return (f'{merge_results.rows_inserted} record(s) inserted, {merge_results.rows_updated} record(s) updated in the INSURANCE_GOLD table')

                                except Exception as e:
                                        return (f'Error with writing results to gold table: {e}')
                except Exception as e:
                        telemetry.add_event("pipeline_failure", {
                                "error": str(e),
                                "stack_trace": traceback.format_exc()
                        })
                        raise  # Re-raise to preserve error handling