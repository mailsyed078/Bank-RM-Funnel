# Databricks notebook source
#importing the dataset"
data = spark.table('rm_funnel.combined_data_v_1_2')
data.display()

# COMMAND ----------

# Creating new data by dropping columns
df = data.drop('Lead_ID',
 'Customer_ID','Lead_Owner', 'Lead_Generated',
 'Lead_Closed',
 'Lead_Closed_In_Days','First_Name',
 'Last_Name','Education_Level', 'Touchpoint',
 'Email_Open_Rate',
 'Click-Through_Rate_on_open_email',
 'Website_Visits',
 'Previous_Product_Type', 'Start_Date',
 'Closure_Date',
 'Tenure_in_Months',
 'Interest_Rates',
 'EMI','Payment_Method',
 'Referral_Channel',
 'Device_Type')
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest Classification 

# COMMAND ----------

import category_encoders
from category_encoders.target_encoder import TargetEncoder

# COMMAND ----------

import pandas as pd
pdf = df.toPandas()
X = pdf.drop('Conversion_Status',axis = 1)
y = pd.get_dummies(pdf['Conversion_Status'],drop_first=True)


# COMMAND ----------

X.columns

# COMMAND ----------

num_cols = X.select_dtypes(include='number').columns
cat_cols = X.select_dtypes(exclude='number').columns
print(num_cols)
print(cat_cols)

# COMMAND ----------

enc_tar = TargetEncoder(cols = ['Lead_Source ', 'Account_Type', 'Product/Service_Interest', 'Location',
       'City', 'Gender', 'Maritial_Status', 'Property_Ownership', 'Profession',
       'Loan_Status'])
enc_df = enc_tar.fit_transform(X,y)

# COMMAND ----------

enc_df

# COMMAND ----------

concated_df = pd.concat([enc_df,y],axis=1).rename(columns = {'Not Converted':'Conversion_Status'})
spark_df = spark.createDataFrame(concated_df)
spark_df.display()

# COMMAND ----------

## Importing required libraries
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

#assembling the columns:
vc = VectorAssembler(inputCols = spark_df.drop('Conversion_Status').columns,outputCol='inp_features')

# COMMAND ----------

## Spilliting data
train, val, test = spark_df.randomSplit([0.6,0.2,0.2])

# COMMAND ----------

# creating model and Pipeline
from pyspark.ml import Pipeline
rf = RandomForestClassifier(featuresCol='inp_features',
                        labelCol='Conversion_Status')
stages = [vc,rf]
pipeline_rf = Pipeline(stages = stages)

# COMMAND ----------

# training the model with default hyperparameter
pipe_model = pipeline_rf.fit(train)
preds_rf = pipe_model.transform(val)
preds_rf.display()

# COMMAND ----------

# Model evaluation
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(metricName ='f1',labelCol ='Conversion_Status')
score = evaluator.evaluate(preds_rf)
print(score)

# COMMAND ----------

help(RandomForestClassifier)

# COMMAND ----------

# DBTITLE 1,Hyperparameter Tuning for Random Forest model
# defining objective_function
import mlflow

def objective_fn(params):
    # setting the hyperparamters
    max_depth = params["max_depth"]
    num_trees = params["num_trees"]

    with mlflow.start_run(run_name = "lr_model_hyp") as run:
        estimator = pipeline_rf.copy({rf.maxDepth:max_depth,
                                      rf.numTrees:num_trees})
        model = estimator.fit(train)
        preds = model.transform(val)
        score = evaluator.evaluate(preds)
        mlflow.log_metric('f1_score',score) 

        return -score

# COMMAND ----------

# defining the search space
from hyperopt import hp

search_space = {
    'max_depth': hp.choice('max_depth',[5,10,15]), 
    'num_trees': hp.choice('num_trees',[100,200,500,700,1000])  
}


# COMMAND ----------

# creating the best_parameters
from hyperopt import hp,tpe, Trials, fmin
best_params = fmin(fn=objective_fn,
                   space = search_space,
                   algo = tpe.suggest,
                   max_evals = 15,
                   trials = Trials())
best_params

# COMMAND ----------

# Retraining the LR model with the best parameter
with mlflow.start_run(run_name = 'LR_final_model') as run:
    mlflow.autolog()

    best_max_iter = best_params['max_iter']
    best_threshold = best_params['threshold']

    # creating pipeline
    estimator = pipeline_lr.copy({lr.maxIter:best_max_iter, lr.threshold:best_threshold})
    model = estimator.fit(train)
    preds = model.transform(val)
    score = evaluator.evaluate(preds)
    
    # logging the parameters and metrics
    
    mlflow.log_metrics('f1_score',score) 
    mlflow.log_param('best_max_iter',best_max_iter)
    mlflow.log_param('best_threshold',best_threshold)


# COMMAND ----------

# DBTITLE 1,Prediction on test data.
import mlflow

run_id = run.info.run_id
logged_model = f'runs:/{run_id}/model'

#Loading model
loaded_model = mlflow.spark.load_model(logged_model)

# Performing inference from the loaded model
predictions = loaded_model.transform(test)
predictions.display()

# COMMAND ----------

#accuracy on test data:
score = evaluator.evaluate(predictions)
print(score)
