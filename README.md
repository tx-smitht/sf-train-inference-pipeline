# sf-train-inference-pipeline

This repo contains the code to build the pipeline described in this [Medium article](https://medium.com/@thomasw-smith/automated-training-and-inference-pipeline-in-snowflake-w-model-registry-7fefd76636c2): 

It is assumed that you have some exposure to Snowflake and machine learning. 

The "create_in_snowflake" folder contains all the code necessary to create the pipeline completely using the Snowflake UI. 

If you'd like to run the code locally, you can use the code in "create_locally". Ensure that your environment is set up to work with Snowpark for Python. If you need help, look [here](https://docs.snowflake.com/en/developer-guide/snowpark/python/setup)! I suggest creating a conda environment based on the conda_env.yml file I've borrowed from the [Snowflake's Snowpark ML Quickstart](https://github.com/Snowflake-Labs/sfguide-intro-to-machine-learning-with-snowpark-ml-for-python/tree/main).

The code helps to create this pipeline, excluding the data coming in from Snowpipe Streaming.

![ml pipeline](https://github.com/tx-smitht/sf-train-inference-pipeline/assets/112910116/96d3420f-8870-4ae9-8356-f7d1a9d0b870)

If you want to do this yourself, start with creating the necessary tables (sql file), then load the data (data_load.ipynb), then run through the model training and prediction python files. 

I used the popular [Health Insurance dataset](https://www.kaggle.com/datasets/sridharstreaks/insurance-data-for-machine-learning?resource=download) and loaded it in using Kaggle's API 
```
kaggle datasets download -d sridharstreaks/insurance-data-for-machine-learning --unzip
```


Here's a gif of the Streamlit app that lets you see your models and how they're performing:
![Mar-19-2024 17-15-47](https://github.com/tx-smitht/sf-train-inference-pipeline/assets/112910116/19704cbe-0116-4e9e-9e4d-03499d2c26f6)
