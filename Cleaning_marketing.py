# Databricks notebook source
file_path = '/FileStore/Marketing_Automation_V__2_0.csv'

# COMMAND ----------

raw_df = spark.read.csv(file_path, header="true", inferSchema="true")
display(raw_df)

# COMMAND ----------

raw_df.count(),len(raw_df.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dropping columns

# COMMAND ----------

raw_df.columns

# COMMAND ----------

raw_df = raw_df.drop('Device_Type')
raw_df.columns

# COMMAND ----------

raw_df.select('_c9').distinct().display()

# COMMAND ----------

raw_df = raw_df.drop('_c9')
raw_df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### checking and rectifying datatypes

# COMMAND ----------

raw_df.printSchema()

# COMMAND ----------

### all the datatypes are correct in their places

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stats

# COMMAND ----------

dbutils.data.summarize(raw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Working on duplictes

# COMMAND ----------

raw_df.dropDuplicates().count()

# COMMAND ----------

df = raw_df.dropDuplicates()
df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### there are no outleirs and null values

# COMMAND ----------

df.columns

# COMMAND ----------

df = df.withColumnsRenamed({'Email_Open_Rate(%)':'Email_open_rate','Click-Through_Rate_on_open_email_(%)':'click_through_rate'})
df.columns

# COMMAND ----------

df.write.mode('overwrite').saveAsTable('marketing_rm_funnel')
