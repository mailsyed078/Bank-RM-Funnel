# Databricks notebook source
crm = spark.table('crm_data_rmfunnel')
display(crm)

# COMMAND ----------

ext = spark.table('ext_data_rmfunnel')
ext.display()

# COMMAND ----------

sales = spark.table('sales_transaction_rm_funnel')
display(sales)

# COMMAND ----------

market = spark.table('marketing_rm_funnel')
display(market)

# COMMAND ----------

df1 = crm.join(ext,on ='Customer_ID').join(market, on ='Customer_ID').join(sales,on="Customer_ID")

# COMMAND ----------

# MAGIC %sql create database rm_funnel

# COMMAND ----------

# MAGIC %sql use rm_funnel

# COMMAND ----------

df1.write.mode('overwrite').saveAsTable('funnel_final_data')
