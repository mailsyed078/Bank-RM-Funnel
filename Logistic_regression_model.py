# Databricks notebook source
df = spark.table('rm_funnel.rm_funnel_combined_data_v_1_1')
df.display()

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

help(LogisticRegression)

# COMMAND ----------

# creating model and Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol=''


