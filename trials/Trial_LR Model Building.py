# Databricks notebook source
#importing the dataset"
data = spark.table('rm_funnel.combined_data_v_1_2')
data.display()

# COMMAND ----------

## checking the shape of data
data.count(),len(data.columns)

# COMMAND ----------

#checking for duplicates
data.dropDuplicates().count()

# COMMAND ----------

# Statistical analysis
dbutils.data.summarize(data)

# COMMAND ----------

data.columns

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

#target encoding the values:
import category_encoders
from category_encoders.target_encoder import TargetEncoder

# COMMAND ----------

import pandas as pd
pd_df = df.toPandas()
X = pd_df.drop('Conversion_Status',axis=1)
y = pd.get_dummies(pd_df['Conversion_Status'],drop_first=True)
tenc = TargetEncoder(cols=['Location','Profession'])
enc_df = tenc.fit_transform(X,y)

# COMMAND ----------

enc_df = pd.concat([enc_df,y],axis=1).rename(columns={'Not Converted':"Conversion_Status"})
spark_df = spark.createDataFrame(enc_df)
spark_df.display()

# COMMAND ----------

## Spilliting data
train, val, test = spark_df.randomSplit([0.6,0.2,0.2])

# COMMAND ----------

# Scaling and converting categorical into numnerical
from pyspark.ml.feature import StandardScaler,RFormula
rform = RFormula(formula ='Conversion_Status ~ .',
                 featuresCol ='enc_features',
                 labelCol ='Conversion_Status_enc',
                 handleInvalid = 'skip')
sc = StandardScaler(inputCol ='enc_features',
                    outputCol = 'sc_features')


# COMMAND ----------

# applying r forumla for the train data
rform_model = rform.fit(train)
enc_df_train = rform_model.transform(train)
enc_df_train.display()

# COMMAND ----------

# applying standard scaler on train
sc_model_train = sc.fit(enc_df_train)
sc_df_train = sc_model_train.transform(enc_df_train)
sc_df_train.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression 

# COMMAND ----------

# creating model and Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol='sc_features',
                        labelCol='Conversion_Status_enc', maxIter=1680,
                        threshold=0.5)
stages = [rform,sc,lr]
pipeline_lr = Pipeline(stages = stages)

# COMMAND ----------

# training the model with default hyperparameter
pipe_model = pipeline_lr.fit(train)
preds_lr = pipe_model.transform(val)
preds_lr.display()

# COMMAND ----------

ratio_df =  preds_lr.groupBy('Prediction').count().toPandas()
ratio_df

# COMMAND ----------

effectiveness = (ratio_df['count'][0]/(ratio_df['count'][0]+ratio_df['count'][1]))*100
effectiveness

# COMMAND ----------

# Model evaluation
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator_lr = MulticlassClassificationEvaluator(metricName ='f1',labelCol ='Conversion_Status_enc')
score_lr = evaluator_lr.evaluate(preds_lr)
print(score_lr)

# COMMAND ----------

# DBTITLE 1,Hyperparameter Tuning for Logistic Regression model
# defining objective_function
import mlflow

def objective_fn(params):
    # setting the hyperparamters
    max_iter = params["max_iter"]
    threshold = params['threshold']

    with mlflow.start_run(run_name = "lr_model_hyp") as run:
        estimator = pipeline_lr.copy({lr.maxIter:max_iter,
                                      lr.threshold:threshold})
        model = estimator.fit(train)
        preds = model.transform(val)
        score = evaluator_lr.evaluate(preds)
        mlflow.log_metric('f1_score',score) 

        return -score

# COMMAND ----------

# defining the search space
from hyperopt import hp

search_space = {
    'max_iter': hp.quniform('max_iter',100,2000,10), 
    'threshold': hp.uniform('threshold',0.1,0.9)  
}


# COMMAND ----------

# creating the best_parameters
from hyperopt import hp,tpe, Trials, fmin
best_params = fmin(fn=objective_fn,
                   space = search_space,
                   algo = tpe.suggest,
                   max_evals = 32,
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



# COMMAND ----------

#accuracy on test data:
score = evaluator.evaluate(predictions)
print(score)
