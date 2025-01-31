-- This SQL command creates a stored procedure identical to the one created in the local folder of the GitHub repo (train_and_save_model.py)
create or replace procedure train_save_ins_model(source_of_truth VARCHAR, major_version BOOLEAN)
    returns String
    language python
    runtime_version = 3.11
    packages =(
        'opentelemetry-api',
        'snowflake-ml-python',
        'snowflake-snowpark-python',
        'snowflake-telemetry-python',
        'xgboost'
    )
    handler = 'train_save_ins_model'
    as '        
# Imports and create a session
from snowflake.snowpark import Session
import snowflake.snowpark
from snowflake.snowpark.dataframe import col
import snowflake.snowpark.functions as F
from snowflake.snowpark.functions import sproc
import snowflake.snowpark.types as T
from snowflake import telemetry

import json
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

def train_save_ins_model(session: Session, source_of_truth: str, major_version: bool = True) -> str:

    tracer = trace.get_tracer("insurance_model.train")

    with tracer.start_as_current_span("train_save_ins_model") as main_span:
        try:
            
            telemetry.set_span_attribute("model.name", "INSURANCE_CHARGES_PREDICTION")

            # Data loading
            with tracer.start_as_current_span("data_loading"):

                # Access the data from the source of truth table
                try:
                    df = session.table(source_of_truth).limit(10000)
                    telemetry.set_span_attribute("data.row_count", df.count())

                except Exception as e:
                    return (f''Error with getting table data: {e}'')

            # Feature engineering
            with tracer.start_as_current_span("feature_engineering"):
                # Define label and feature columns
                LABEL_COLUMNS = [''CHARGES'']
                FEATURE_COLUMN_NAMES = [i for i in df.schema.names if i not in LABEL_COLUMNS]
                OUTPUT_COLUMNS = [''PREDICTED_CHARGES'']

                # Define Snowflake numeric types (possibly for scaling, ordinal encoding)
                # numeric_types = [T.DecimalType, T.DoubleType, T.FloatType, T.IntegerType, T.LongType]
                # numeric_columns = [col.name for col in df.schema.fields if (type(col.datatype) in numeric_types) and (col.name in FEATURE_COLUMN_NAMES)]

                # Define Snowflake categorical types and determine which columns to OHE
                categorical_types = [T.StringType]
                cols_to_ohe = [col.name for col in df.schema.fields if (type(col.datatype) in categorical_types)]
                ohe_cols_output = [col + ''_OHE'' for col in cols_to_ohe]


                # Standardize the values in the rows by removing spaces, capitalizing
                def fix_values(columnn):
                        return F.upper(F.regexp_replace(F.col(columnn), ''[^a-zA-Z0-9]+'', ''_''))

                try:
                    for col in cols_to_ohe:
                            df = df.na.fill(''NONE'', subset=col)
                            df = df.withColumn(col, fix_values(col))
                    telemetry.add_event("feature_engineering_complete")

                except Exception as e:
                    return (f''Error with standardizing values: {e}'')

            # Model training
            with tracer.start_as_current_span("define_pipeline"):
                # Define the pipeline
                try:
                    pipe = Pipeline(
                        steps=[
                            #(''imputer'', SimpleImputer(input_cols=all_cols)),
                            #(''mms'', snowmlpp.MinMaxScaler(input_cols=cols_to_scale, output_cols=scale_cols_output)),
                            (''ohe'', snowmlpp.OneHotEncoder(input_cols=cols_to_ohe, output_cols=ohe_cols_output, drop_input_cols=True)),
                            (''grid_search_reg'', GridSearchCV(estimator=XGBRegressor(),
                                                                param_grid={ "n_estimators":[50, 100, 200], # 25
                                                                            "learning_rate":[0.01, 0.1, 0.5 ], # .5
                                                                            },
                                                                n_jobs = -1,
                                                                scoring="neg_mean_absolute_percentage_error",
                                                                input_cols=FEATURE_COLUMN_NAMES.append(ohe_cols_output),
                                                                label_cols=LABEL_COLUMNS,
                                                                output_cols=OUTPUT_COLUMNS
                                                                )
                            )
                        ]      
                    )
                    

                except Exception as e:
                    return (f''Error with defining the pipeline: {e}'')

            with tracer.start_as_current_span("train_test_split"):
                # Split the data into training and testing
                train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

            # Fit the pipeline
            with tracer.start_as_current_span("fit_pipeline"):
                try:
                    pipe.fit(train_df)
                    telemetry.set_span_attribute("training.param_grid", "Fitting done")
                
                except Exception as e:
                    return (f''Error with fitting pipeline: {e}'')


            # Model evaluation
            with tracer.start_as_current_span("model_evaluation"):
                # Predict with the pipeline
                try:
                    results = pipe.predict(test_df)

                except Exception as e:
                    return (f''Error with predicting with pipeline: {e}'')


                # Use Snowpark ML metrics to calculate MAPE and MSE

                # Calculate MAPE
                mape = mean_absolute_percentage_error(df=results, y_true_col_names=LABEL_COLUMNS, y_pred_col_names=OUTPUT_COLUMNS)

                # Calculate MSE
                mse = mean_squared_error(df=results, y_true_col_names=LABEL_COLUMNS, y_pred_col_names=OUTPUT_COLUMNS)
                telemetry.set_span_attribute("model.mape", mape)
                telemetry.set_span_attribute("model.mse", mse)

            # Model registration
            with tracer.start_as_current_span("model_registration"):
                def set_model_version(registry_object,model_name, major_version=True):
                    # See what we''ve logged so far, dynamically set the model version
                    import numpy as np
                    import json
                    
                    model_list = registry_object.show_models()
                    
                    if len(model_list) == 0:
                        return ''V1''
                    
                    model_list_filter = model_list[model_list[''name''] ==  model_name]

                    if len(model_list_filter) == 0:
                        return ''V1''

                    version_list_string = model_list_filter[''versions''].iloc[0]
                    version_list = json.loads(version_list_string)
                    version_numbers = [float(s.replace(''V'', '''')) for s in version_list]
                    model_last_version = max(version_numbers)
                    
                    
                    if np.isnan(model_last_version) == True:
                        model_new_version = ''V1''

                    elif np.isnan(model_last_version) == False and major_version == True:
                        model_new_version = round(model_last_version + 1,2)
                        model_new_version = ''V'' + str(model_new_version)
                        
                    else:
                        model_new_version = round(model_last_version + .1,2)
                        model_new_version = ''V'' + str(model_new_version)
                        
                    return model_new_version # This is the version we will use when we log the new model.

                # Create model regisry object
                try:
                    model_registry = registry.Registry(session=session, database_name=session.get_current_database(), schema_name=''ML_PIPE'')

                except Exception as e:
                    return (f''Error with creating model registry object: {e}'')
                
                # Save model to registry
                try:
                    LABEL_COLUMNS = [''CHARGES'']
                    FEATURE_COLUMN_NAMES = [i for i in df.schema.names if i not in LABEL_COLUMNS]
                    X = train_df.select(FEATURE_COLUMN_NAMES).limit(100)

                    model_name = ''INSURANCE_CHARGES_PREDICTION''
                    version_name = set_model_version(model_registry, model_name, major_version=major_version)
                    model_version = model_registry.log_model(
                        model = pipe, 
                        model_name = model_name, 
                        version_name= f''"{version_name}"'',
                        sample_input_data=X,
                        conda_dependencies=[''snowflake-snowpark-python'',''snowflake-ml-python'',''scikit-learn'', ''xgboost'']
                        )

                    model_version.set_metric(metric_name=''mean_abs_pct_err'', value=mape)
                    model_version.set_metric(metric_name=''mean_sq_err'', value=mse)
                    telemetry.add_event("model_registered", {"version": version_name})
                
                except Exception as e:
                    return (f''Error with saving model to registry: {e}'')
                
                try:
                    session.sql(f''alter model INSURANCE_CHARGES_PREDICTION set default_version = "{version_name}";'')
                
                except Exception as e:
                    return (f''Error with setting default version: {e}'')

            return f''Model {model_name} has been logged with version {version_name} and has a MAPE of {mape} and MSE of {mse}''

        except Exception as e:
            telemetry.add_event("pipeline_failure", {
                "error": str(e),
                "stack_trace": traceback.format_exc()
            })
            raise  # Re-raise to preserve error handling
';