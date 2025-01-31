CREATE DATABASE INSURANCE;
CREATE SCHEMA ML_PIPE;


-- Create the table that we will use as our source of truth to train on
CREATE or REPLACE TABLE INSURANCE.ML_PIPE.SOURCE_OF_TRUTH (
	AGE NUMBER(38,0),
	GENDER VARCHAR(16777216),
	BMI FLOAT,
	CHARGES FLOAT,
	CHILDREN NUMBER(38,0),
	SMOKER VARCHAR(16777216),
	REGION VARCHAR(16777216),
	MEDICAL_HISTORY VARCHAR(16777216),
	FAMILY_MEDICAL_HISTORY VARCHAR(16777216),
	EXERCISE_FREQUENCY VARCHAR(16777216),
	OCCUPATION VARCHAR(16777216),
	COVERAGE_LEVEL VARCHAR(16777216)
);

-- * Run the data load notebook either locally or in snowflake. It inserts 10k rows into the SOURCE_OF_TRUTH table. This serves as the data to train the model.

-- Create table that will hold the remaining 990k records to be inserted into the landing table, simulating streamed-in data.
CREATE or REPLACE TABLE INSURANCE.ML_PIPE.INCOMING_DATA_SOURCE (
	AGE NUMBER(38,0),
	GENDER VARCHAR(16777216),
	BMI FLOAT,
	CHARGES FLOAT,
	CHILDREN NUMBER(38,0),
	SMOKER VARCHAR(16777216),
	REGION VARCHAR(16777216),
	MEDICAL_HISTORY VARCHAR(16777216),
	FAMILY_MEDICAL_HISTORY VARCHAR(16777216),
	EXERCISE_FREQUENCY VARCHAR(16777216),
	OCCUPATION VARCHAR(16777216),
	COVERAGE_LEVEL VARCHAR(16777216)
);
-- Again, I loaded this table through Snowpark in the load_data.ipynb file.

-- Create a stage to hold the SPROCs
CREATE STAGE ML_PIPE_STAGE;

-- * Run the train_model.py file before moving on. This will create the training sproc. * --

CREATE OR REPLACE EVENT TABLE INSURANCE.ML_PIPE.MODEL_TRACES;
ALTER ACCOUNT SET EVENT_TABLE = INSURANCE.ML_PIPE.MODEL_TRACES;

-- Create the task that calls the training sproc
CREATE or REPLACE TASK TRAIN_SAVE_TASK
  WAREHOUSE = ML_WAREHOUSE -- Replace with your warehouse
  SCHEDULE = '11520 MINUTE' -- Executes every 8 days 
  AS
    CALL TRAIN_SAVE_INS_MODEL('SOURCE_OF_TRUTH',FALSE);

-- Tasks are created in a suspended state. Resume it
ALTER TASK TRAIN_SAVE_TASK RESUME;

-- Execute immediately so that you have a trained model in registry
EXECUTE TASK TRAIN_SAVE_TASK;

use warehouse ml_warehouse;
call train_save_ins_model_cloud('SOURCE_OF_TRUTH',FALSE);
USE WAREHOUSE COMPUTE_WH;

-- Create the landing table (where streamed-in records could land)
CREATE or REPLACE TABLE INSURANCE.ML_PIPE.LANDING_TABLE (
	AGE NUMBER(38,0),
	GENDER VARCHAR(16777216),
	BMI FLOAT,
	CHARGES FLOAT,
	CHILDREN NUMBER(38,0),
	SMOKER VARCHAR(16777216),
	REGION VARCHAR(16777216),
	MEDICAL_HISTORY VARCHAR(16777216),
	FAMILY_MEDICAL_HISTORY VARCHAR(16777216),
	EXERCISE_FREQUENCY VARCHAR(16777216),
	OCCUPATION VARCHAR(16777216),
	COVERAGE_LEVEL VARCHAR(16777216)
);

-- Create the stream on the landing table
CREATE OR REPLACE STREAM STREAM_ON_LANDING ON TABLE LANDING_TABLE;

-- Create a gold table for the records and their predictions to land
CREATE OR REPLACE TABLE INSURANCE_GOLD(
    AGE NUMBER(38,0),
	GENDER VARCHAR(16777216),
	BMI FLOAT,
	CHILDREN NUMBER(38,0),
	SMOKER VARCHAR(16777216),
	REGION VARCHAR(16777216),
	MEDICAL_HISTORY VARCHAR(16777216),
	FAMILY_MEDICAL_HISTORY VARCHAR(16777216),
	EXERCISE_FREQUENCY VARCHAR(16777216),
	OCCUPATION VARCHAR(16777216),
	COVERAGE_LEVEL VARCHAR(16777216),
    METADATA$ROW_ID VARCHAR(16777216),
    METADATA$ISUPDATE BOOLEAN,
    METADATA$ACTION VARCHAR(16777216),
    METADATA_UPDATED_AT DATE,
    CHARGES FLOAT,
    PREDICTED_CHARGES FLOAT
);

-- Insert records into the landing table to simulate streamed data
INSERT INTO LANDING_TABLE(
    AGE ,
	GENDER,
	BMI,
	CHARGES ,
	CHILDREN,
	SMOKER,
	REGION,
	MEDICAL_HISTORY ,
	FAMILY_MEDICAL_HISTORY,
	EXERCISE_FREQUENCY ,
	OCCUPATION ,
	COVERAGE_LEVEL
) SELECT 
    AGE,
	GENDER,
	BMI,
	CHARGES ,
	CHILDREN,
	SMOKER,
	REGION,
	MEDICAL_HISTORY ,
	FAMILY_MEDICAL_HISTORY,
	EXERCISE_FREQUENCY ,
	OCCUPATION ,
	COVERAGE_LEVEL
FROM INCOMING_DATA_SOURCE
LIMIT 500000; -- Change this number to test prediction speed at different quantities

-- View the inserted records in the stream, along with the added metadata columns
SELECT * FROM STREAM_ON_LANDING;

-- * Run the predict_and_write.py file here. This will create the prediction/write to gold sproc * --

-- Call the prediction SPROC to see it work on the data you loaded into the stream.
USE WAREHOUSE ML_WAREHOUSE;
CALL PREDICT_WRITE_TO_GOLD();
USE WAREHOUSE COMPUTE_WH;

-- Testing the capacity to update records that already exist in the gold table
update landing_table set coverage_level = 'STANDARD'
where age = 41;

-- View the stream after updating
SELECT * FROM STREAM_ON_LANDING;

-- Call the prediction SPROC again to see how it handles updates
CALL PREDICT_WRITE_TO_GOLD();

-- Create the predict and write task
CREATE or REPLACE TASK PREDICT_WRITE_TASK
  WAREHOUSE = COMPUTE_WH
  SCHEDULE = '1 MINUTE'
  WHEN
    SYSTEM$STREAM_HAS_DATA('STREAM_ON_LANDING')
  AS
    CALL PREDICT_WRITE_TO_GOLD();

-- Again, tasks are created in a suspended state. Resume it
ALTER TASK PREDICT_WRITE_TASK RESUME;

-- Clean up
ALTER TASK PREDICT_WRITE_TASK SUSPEND;
ALTER TASK TRAIN_SAVE_TASK SUSPEND;

-- -- Clean up
-- DROP DATABASE INSURANCE;
