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
df = data.drop()
df.display()

# COMMAND ----------

## Spilliting data
train, val, test = df.randomSplit([0.6,0.2,0.2])

# COMMAND ----------

### I got passed the data
# Scaling and converting categorical into numnerical
from pyspark.ml.feature import StandardScaler,RFormula
rform = RFormula(formula ='Conversion_Status ~.',
                 featuresCol ='enc_features',
                 labelCol ='Conversion_Status',
                 handleInvalid = 'skip')
sc = StandardScaler(inputCol ='enc_features',
                    outputCol = 'sc_features')


# COMMAND ----------

# applying r foruma for the train data
rform_model = rform.fit(train)
enc_df_train = rform_model.transfrom(train)
enc_df_train.display()

# COMMAND ----------

#applying r formula for the val data
rform_model = rform.fit(val)
enc_df_val = rform_model.transfrom(val)
enc_df_val.display()


# COMMAND ----------

#applying r formula for the test data
rform_model = rform.fit(test)
enc_df_test = rform_model.transfrom(test)
enc_df_test.display()

# COMMAND ----------

# applying standard scaling on train, val and test
sc_model_train = sc.fit(enc_df_train)
sc_df_train = sc_model_train.transform(enc_df_train)
sc_df_train.display()

# COMMAND ----------

# applying standard scaling on val 
sc_model_val = sc.fit(enc_df_val)
sc_df_val = sc_model_val.transform(enc_df_val)
sc_df_val.display()

# COMMAND ----------

# applying standard scaling on test
sc_model_test = sc.fit(enc_df_test)
sc_df_test = sc_model_test.transform(enc_df_test)
sc_df_test.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression 

# COMMAND ----------

# creating model and Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol='sc_features',
                        labelCol='Conversion_Status')
stages = [rform,sc,lr]
pipeline_lr = Pipeline(stages = stages)

# COMMAND ----------

# training the model with default hyperparameter
pipe_model = pipeline_lr.fit(train)
preds_lr = pipe_model.transform(val)
preds_lr.display()

# COMMAND ----------

# Model evaluation
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator_lr = MulticlassClassificationEvaluator(metricName ='f1',lableCol ='Conversion_Status')
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
        score = evaluator.evaluate(preds)
        mlflow.log_metrics('f1_score',score) 

        return -score

# COMMAND ----------

# defining the search space
from hyperopt import hp

search_space = {
    'max_iter': hp.quniform('max_iter',100,500,1), 
    'threshold': hp.quniform('threshold',0.1,1,1)  
}


# COMMAND ----------

# creating the best_parameters
from hyperopt import hp,tpe, Trials, fmin
best_params = fmin(fn=objective_fn,
                   space = search_space,
                   algo = tpe.suggest,
                   max_evals = 4,
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
