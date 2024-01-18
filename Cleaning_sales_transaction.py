# Databricks notebook source
file_path = '/FileStore/Sales___Transaction__Historical_data__V__2.csv'

# COMMAND ----------

raw_df = spark.read.csv(file_path, header="true", inferSchema="true", multiLine="true", escape='"')
display(raw_df)

# COMMAND ----------

raw_df.count(),len(raw_df.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dropping columns

# COMMAND ----------

raw_df.columns

# COMMAND ----------

raw_df = raw_df.drop('Order_ID')
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

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### there are no outleirs and null values
