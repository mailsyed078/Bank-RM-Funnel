# Databricks notebook source
#importing the dataset"
data = spark.table('rm_funnel.combined_data_v_1_2')
data.display()

# COMMAND ----------

## checking the shape of data
data.count(),len(data.columns)

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
# MAGIC ### XGB Classification 

# COMMAND ----------

import category_encoders
from category_encoders.target_encoder import TargetEncoder

# COMMAND ----------

import pandas as pd
pdf = df.toPandas()
X = pdf.drop('Conversion_Status',axis = 1)
y = pd.get_dummies(pdf['Conversion_Status'],drop_first=True)


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

concated_df = pd.concat([enc_df,y],axis=1).rename(columns = {'Not Converted':'Conversion_Status'})
spark_df = spark.createDataFrame(concated_df)
spark_df.display()

# COMMAND ----------

!pip install sparkxgb

# COMMAND ----------

## Importing required libraries
import sparkxgb
from sparkxgb.xgboost import XGBoostClassifier
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
xgb = XGBoostClassifier(objective="binary:logistic",
                        featuresCol="inp_features",
                        labelCol="Conversion_Status")
stages = [vc,xgb]
pipeline_xgb = Pipeline(stages = stages)

# COMMAND ----------

# training the model with default hyperparameter
pipe_model = pipeline_xgb.fit(train)
preds_xgb = pipe_model.transform(val)
preds_xgb.display()

# COMMAND ----------

# Model evaluation
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(metricName ='f1',labelCol ='Conversion_Status')
score = evaluator.evaluate(preds_xgb)
print(score)

# COMMAND ----------

help(XGBoostClassifier)

# COMMAND ----------

# DBTITLE 1,Hyperparameter Tuning for XGB model
# defining objective_function
import mlflow

def objective_fn(params):
    # setting the hyperparamters
    max_depth = params["max_depth"]
    alpha = params["alpha"]
    eta = params["eta"]
    eval_metric = params["eval_metric"]


    with mlflow.start_run(run_name = "xgb_model_hyp") as run:
        estimator = pipeline_xgb.copy({xgb.maxDepth:max_depth,
                                      xgb.alpha:alpha,
                                      xgb.eta:eta,
                                      xgb.evalMetric:eval_metric})
        model = estimator.fit(train)
        preds = model.transform(val)
        score = evaluator.evaluate(preds)
        mlflow.log_metric('f1_score',score) 

        return -score

# COMMAND ----------

# defining the search space
from hyperopt import hp
max_depth = [5,7,10]
alpha = [0.01,0.05,0.1]
eta = [0.1,0.2,0.3]
eval_metric = ['error','logloss']

search_space = {
    'max_depth': hp.choice('max_depth',max_depth), 
    'alpha': hp.choice('alpha',alpha),
     'eta': hp.choice('eta',eta),
     'eval_metric':hp.choice('eval_metric',eval_metric)  
}


# COMMAND ----------

# creating the best_parameters
from hyperopt import hp,tpe, Trials, fmin
best_params = fmin(fn=objective_fn,
                   space = search_space,
                   algo = tpe.suggest,
                   max_evals = 54,
                   trials = Trials())
best_params

# COMMAND ----------

# Retraining the LR model with the best parameter
with mlflow.start_run(run_name = 'xgb_final_model') as run:
    mlflow.autolog()

    best_max_depth = max_depth[best_params['max_depth']]
    best_alpha = alpha[best_params['alpha']]
    best_eta = eta[best_params['eta']]
    best_eval_metric = eval_metric[best_params['eval_metric']]

    # creating pipeline
    estimator = pipeline_xgb.copy({xgb.maxDepth:best_max_depth,
                                      xgb.alpha:best_alpha,
                                      xgb.eta:best_eta,
                                      xgb.evalMetric:best_eval_metric})
    model = estimator.fit(train)
    preds = model.transform(val)
    score = evaluator.evaluate(preds)
    mlflow.spark.log_model(model,'model')
    
    # logging the parameters and metric
    mlflow.log_metric('f1_score', score) 
    mlflow.log_param('best_max_depth', best_max_depth)
    mlflow.log_param('best_alpha', best_alpha)
    mlflow.log_param('best_eta', best_eta)
    mlflow.log_param('best_eval_metric', best_eval_metric)

# COMMAND ----------

# DBTITLE 1,Prediction on test data.
# Performing inference from the trained model where model ouput 0 states 'Converted' and 1 states 'Not Converted'
predictions = model.transform(test)
predictions.display()

# COMMAND ----------

#accuracy on test data:
score = evaluator.evaluate(predictions)
print(score)

# COMMAND ----------

#calculating the ratio of conversion
ratio_df =  predictions.groupBy('Prediction').count().toPandas()
ratio_df

# COMMAND ----------

#calculating the effectiveness
effectiveness = (ratio_df['count'][0]/(ratio_df['count'][0]+ratio_df['count'][1]))*100
print(f"As per the given data the effectiveness of the funnel is: {effectiveness}")

# COMMAND ----------


