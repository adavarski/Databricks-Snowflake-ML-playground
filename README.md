## Databricks (ML/DeepML and MLFLow)  

Databricks Unified Analytics Platform (elements):

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Databricks-Unified-Analytics-Platform.png" width="800">

Databricks components:

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Databricks-Unified-analytics-Platform-components-table.png" width="600">

Use MLFlow through the Databricks platform.

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Databrick-UI-overview.png" width="800">

Log in to the Databricks account and spin up a cluster of desired size:

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Databricks-UI-Cluster-UP.png" width="800">

New Notebook cells:

```
dbutils.library.installPyPI('mlflow')
dbutils.library.installPyPI('scikit-learn')
dbutils.library.restartPython()
```
```
import pandas as pd
import sklearn
from sklearn.datasets import load_boston
basedata = load_boston()
pddf = pd.DataFrame(basedata.data
        ,columns=basedata.feature_names)
pddf['target'] = pd.Series(basedata.target)
pct20 = int(pddf.shape[0]*.2)
testdata = spark.createDataFrame(pddf[:pct20])
traindata = spark.createDataFrame(pddf[pct20:])
```
```
from pyspark.ml.feature import VectorAssembler
va = VectorAssembler(inputCols = basedata.feature_names
        ,outputCol = 'features')
testdata = va.transform(testdata)['features','target']
traindata = va.transform(traindata)['features','target']
```
To get some test data for MLflow, we’ll run it three times using a simple loop. We’ll also change the maxItem and regParam hyperparameters to see if they make any difference in the performance of the algorithm. To keep the loop logic to a minimum, we’ll increase both in unison. You wouldn’t do it like this in a real use case:

```
from pyspark.ml.regression import LinearRegression
import mlflow
for i in range(1,4):
  with mlflow.start_run():
    mi = 10 * i
    rp = 0.1 * i
    enp = 0.5
    mlflow.log_param('maxIter',mi)
    mlflow.log_param('regParam',rp)
    mlflow.log_param('elasticNetParam',enp)
    lr = LinearRegression(maxIter=mi
        ,regParam=rp
        ,elasticNetParam=enp
        ,labelCol="target")
    model = lr.fit(traindata)
    pred = model.transform(testdata)
    r = pred.stat.corr("prediction", "target")
    mlflow.log_metric("rsquared", r**2, step=i)
```



Notebook example: 

https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/databrics/Databrics-MLFlow-demo.ipynb

Check MLFlow UI:

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Databrics-MLFlow-experiments.png" width="800">

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Databrics-MLFlow-demo-run.png" width="800">


# Snowflake (Databricks integration)

Snowflake summary: unlimited concurrency for queries, a consolidated DW, and a big data solution on a single data platform, as well as dedicated virtual warehouses for analysts with heavy queries.

Snowflake Architecture:

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Snowflake-architecture.png" width="600">

Snowflake Layers:

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Snowflake-layers.png" width="600">

Snowflake was built from the ground up and designed to handle modern big data and analytics challenges. It combines the benefits of both SMP and MPP architectures and takes full advantage of the cloud.

Similar to an SMP architecture, Snowflake uses a central storage that is accessible from all the compute nodes. In addition, similar to an MPP
architecture, Snowflake processes queries using MPP compute clusters, also known as virtual warehouses. As a result, Snowflake combines the simplicity of data management and scalability with a shared-nothing architecture (like in MPP).

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Snowflake-MPP-vs-SMP.png" width="600">


Note: A specialty of the technical design of the Snowflake is that the data is stored in [micro-partitions](https://docs.snowflake.com/en/user-guide/tables-clustering-micropartitions.html),  which are immutable. This means that with any operations such as the addition or deletion of data, a new
micro-partition is created, and the old one ceases to exist.

Snowflake Architecture (deep):

Shared-Disk vs Shared-Nothing architectures:

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/snareddisk.png" width="600">

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/sharednothing.png" width="600">



What is Snowflake ?

Snowflake is a modern Data Warehouse developed to address issue in existing Data Warehouse tools. It is provided as a Saas (Software-as-a-Service) so we do not need to worry about Hardware or Software maintenance. It enables users to just create tables and start querying data with very less administration or DBA activities needed. Added to that it has an unique and required feature Time Travel.

Snowflake uses hybrid architecture. It is a mixture of both shared disk and shared nothing architecture. For storing data it uses shared disk design where it stores all data in a centralized place that is accessible from all nodes (servers) in the compute cluster. For running/executing query it uses shared nothing design, Snowflake executes query using compute clusters (virtual data warehouse) where each node in the cluster stores a portion of the entire data set locally. This approach offers the data management simplicity of a shared-disk architecture, but with the performance and scale-out benefits of a shared-nothing architecture.

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Snowflake-architecture-full.png" width="800">

As we from the above diagram Snowflake has 3 layers.

    Storage layer (Database storage)
    Compute layer (Query processing)
    Cloud services 

Storage layer (Database storage)

it is similar to shared disk architecture. When we load the data into Snowflake it converts data into COLUMNAR format and store it. Snowflake stores data into multiple micro partitions that are internally optimized and compressed. Since Snowflake Database storage layer uses Cloud based storage, it is elastic and is charged as per the usage per TB every month. (40$ / TB for on-demand and 23$ / TB for pre-purchased storage)

Data storage layer is shared disk architecture and all data warehouse can access it. Compute nodes connect with storage layer to fetch the data for query processing. As the storage layer is independent, we only pay for the average monthly storage used. Snowflake charges only for storing actual data and storing metadata (DB schema, View, etc.,) are free of cost.

Compute Layer (Query Processing)

It is similar to shared nothing architecture. This Layer uses Virtual Warehouse for executing query (DDL and DML) on the data stored.  Snowflake separates the query processing layer from the disk storage. Queries execute in this layer using the data from the storage layer.

Each virtual warehouse is an independent compute cluster and MPP (Massively parallel process) that does not share compute resources with other virtual warehouses. As a result, each virtual warehouse has no impact on the performance of other virtual warehouses.

Each virtual warehouse runs with its own compute and caching. In Snowflake, while queries are running, compute resources can scale without disruption or downtime, and without the need to redistribute/rebalance data (storage). Scaling of compute resources can occur automatically, with auto-sensing. This means the Snowflake software can automatically detect when scaling is needed and scale your environment without admin or user involvement.

Virtual warehouse will be auto-suspended when there is no query to execute and will be auto-resumed when there is a query to run. This is managed by Snowflake. We pay when the Virtual warehouse is active, meaning when we execute query. Query processing (Virtual Warehouse) are charged in the form of Snowflake credits. (I will cover about Snowflake cost in different post).

Cloud Service

This layer provides all necessary functionality that coordinates across Snowflake.This layer also runs on compute instances provisioned by Snowflake from the cloud provider so there is no cost for this service.

This layer provides following services,

    Authentication –  login request is handled in this layer.
    Metadata management – metadata related to query optimization are available in this layer.
    Query parsing and optimization – When a user executed a SQL query that will be parsed and optimized in this layer and then forwarded to Compute Layer for query processing.
    Access control – Role based access managements are handled in this layer.

As we can see here – All this 3 layers are loosely coupled and we can scale any one layer independently of others. We pay for only Storage and Compute layer. 

Conclusion

    Snowflake is a true SaaS cloud data warehouse.
    Snowflake follows hybrid architecture to handle storage and compute.
    In Snowflake all the layers are independent. We can easily auto-scale any of the layers.
    Snowflake uses different pricing model for Storage and Query processing layers.
    We pay for what we store in Storage layer and pay for the amount of query execution time in Query processing layer. 


Planning: Deciding on a Snowflake Edition

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Snowflake-editions.png" width="800">


Creating a Snowflake Account: 

Click Start for Free in the upper-right corner of the page (www.snowflake.com). This will give you a 30-day trial of Snowflake plus 400 Snowflake credits to play with. Enter the following required details: name, company name, e-mail, phone number, Snowflake edition, cloud provider, and region. Finish by clicking Create Account, and in about 15 minutes, you will receive an e-mail with a link to your web interface.

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Snowflake-UI-startup.png" width="800">


Installing Snowsql

SnowSQL is the next-generation command-line client for connecting to Snowflake, executing SQL queries, and performing all DDL and DML operations, including loading data into and unloading data out of database tables.

```
$ curl -O https://sfc-repo.snowflakecomputing.com/snowsql/bootstrap/1.2/linux_x86_64/snowsql-1.2.10-linux_x86_64.bash
$ bash snowsql-1.2.10-linux_x86_64.bash 

```

Configure snowsql (Add your connection information to the ~/.snowsql/config file)
```
$ cat ~/.snowsql/config |grep -v "^#"
[connections]          


accountname = vw60186
username = ADAVARSKI
password = Kr0k0dil!
region = eu-central-1
```

Snowsql (test connection):

```
$ snowsql 
* SnowSQL * v1.2.10
Type SQL statements or !help
ADAVARSKI#COMPUTE_WH@(no database).(no schema)>

$ snowsql -s TPCH_SF001 -d SNOWFLAKE_SAMPLE_DATA
* SnowSQL * v1.2.10
Type SQL statements or !help
ADAVARSKI#COMPUTE_WH@SNOWFLAKE_SAMPLE_DATA.TPCH_SF001>

```

Create DW examples:

```
$ snowsql
CREATE WAREHOUSE DEVELOPMENT WITH WAREHOUSE_SIZE =
'XSMALL' WAREHOUSE_TYPE = 'STANDARD' AUTO_SUSPEND = 600
AUTO_RESUME = TRUE MIN_CLUSTER_COUNT = 1 MAX_CLUSTER_
COUNT = 2 SCALING_POLICY = 'STANDARD';

CREATE WAREHOUSE PRODUCTION WITH WAREHOUSE_SIZE =
'XSMALL' WAREHOUSE_TYPE = 'STANDARD' AUTO_SUSPEND = 600
AUTO_RESUME = TRUE MIN_CLUSTER_COUNT = 1 MAX_CLUSTER_
COUNT = 2 SCALING_POLICY = 'STANDARD';

#~/.snowsql/config
[connections.development]
password=<your password>
warehousename=DEVELOPMENT
[connections.production]
password=<your password>
warehousename=PRODUCTION

$ snowsql
!connect development
!connect production
!exit
!exit
!quit
```

Bulk Data Loading

To get data into a database table, you need to insert it. Insert statements can take a while since they need to be executed one row at a time. Bulk copying can take a large amount of data and insert it into a database all in one batch. The bulk data loading option in Snowflake allows batch loading of data from files that are in cloud storage, like AWS S3. If your data files are not currently in cloud storage, then there is an option to copy the data files from a local machine to a cloud storage staging area before loading them into Snowflake. This is known as Snowflake’s internal staging area. The data files are transmitted from a local machine to an internal, Snowflake-designated, cloud storage staging location and then loaded into tables using the COPY command.

Tip: External tables can be created instead of loading data into Snowflake. This would be useful when only a portion of data is needed.


Bulk Load Locations

Snowflake supports loading data from files staged in any of the following
cloud storage locations, regardless of the cloud platform for your
Snowflake account:

• Snowflake-designated internal storage staging location

• AWS S3, where files can be loaded directly from any
user-supplied S3 bucket

• GCP Cloud Storage, where files can be loaded directly
from any user-supplied GCP Cloud Storage container

• Azure Blob storage, where files can be loaded directly
from any user-supplied Azure container


Data Loading with SnowSQL:

```
CREATE WAREHOUSE DEVELOPMENT WITH WAREHOUSE_SIZE = 'XSMALL' WAREHOUSE_TYPE = 'STANDARD' AUTO_SUSPEND = 600 AUTO_RESUME = TRUE;
USE WAREHOUSE DEVELOPMENT;
CREATE DATABASE AIRBNB;
USE DATABASE AIRBNB;
USE SCHEMA PUBLIC;
CREATE OR REPLACE TABLE "ZIPCODES2000_SNOWSQL" ("ZIPCODE" STRING, "LON" DOUBLE, "LAT" DOUBLE);
put file:///home/davar/Downloads/zips2000.csv @AIRBNB.PUBLIC.%zipcodes2000_snowsql;
copy into zipcodes2000_snowsql from @%zipcodes2000_snowsql file_format = (type = csv field_optionally_enclosed_by='"' SKIP_HEADER = 1);
select count(*) from ZIPCODES2000_SNOWSQL;

```

Example Output:

```
ADAVARSKI#COMPUTE_WH@(no database).(no schema)>CREATE WAREHOUSE DEVELOPMENT WITH WAREHOUSE_SIZE = 'XSMALL' WAREHOUSE_TYPE = 'STANDARD' AUTO_SUSPEND = 600 AUTO_RESUME = TRUE;
+---------------------------------------------+                                 
| status                                      |
|---------------------------------------------|
| Warehouse DEVELOPMENT successfully created. |
+---------------------------------------------+
1 Row(s) produced. Time Elapsed: 0.945s
ADAVARSKI#DEVELOPMENT@(no database).(no schema)>USE WAREHOUSE DEVELOPMENT;
+----------------------------------+                                            
| status                           |
|----------------------------------|
| Statement executed successfully. |
+----------------------------------+
1 Row(s) produced. Time Elapsed: 0.130s
ADAVARSKI#DEVELOPMENT@(no database).(no schema)>CREATE DATABASE AIRBNB;
+---------------------------------------+                                       
| status                                |
|---------------------------------------|
| Database AIRBNB successfully created. |
+---------------------------------------+
1 Row(s) produced. Time Elapsed: 0.207s
ADAVARSKI#DEVELOPMENT@AIRBNB.PUBLIC>use database AIRBNB;
+----------------------------------+                                            
| status                           |
|----------------------------------|
| Statement executed successfully. |
+----------------------------------+
1 Row(s) produced. Time Elapsed: 0.109s
ADAVARSKI#DEVELOPMENT@AIRBNB.PUBLIC>USE SCHEMA PUBLIC;
+----------------------------------+                                            
| status                           |
|----------------------------------|
| Statement executed successfully. |
+----------------------------------+
1 Row(s) produced. Time Elapsed: 0.104s
ADAVARSKI#DEVELOPMENT@AIRBNB.PUBLIC>CREATE OR REPLACE TABLE "ZIPCODES2000_SNOWSQL" ("ZIPCODE" STRING, "LON" DOUBLE, "LAT" DOUBLE);
+--------------------------------------------------+                            
| status                                           |
|--------------------------------------------------|
| Table ZIPCODES2000_SNOWSQL successfully created. |
+--------------------------------------------------+
1 Row(s) produced. Time Elapsed: 0.453s
ADAVARSKI#DEVELOPMENT@AIRBNB.PUBLIC>show tables;
+-------------------------------+----------------------+---------------+-------------+-------+---------+------------+------+-------+----------+----------------+----------------------+-----------------+-------------+
| created_on                    | name                 | database_name | schema_name | kind  | comment | cluster_by | rows | bytes | owner    | retention_time | automatic_clustering | change_tracking | is_external |
|-------------------------------+----------------------+---------------+-------------+-------+---------+------------+------+-------+----------+----------------+----------------------+-----------------+-------------|
| 2020-12-30 02:53:21.318 -0800 | ZIPCODES2000_SNOWSQL | AIRBNB        | PUBLIC      | TABLE |         |            |    0 |     0 | SYSADMIN | 1              | OFF                  | OFF             | N           |
+-------------------------------+----------------------+---------------+-------------+-------+---------+------------+------+-------+----------+----------------+----------------------+-----------------+-------------+
1 Row(s) produced. Time Elapsed: 0.122s
ADAVARSKI#DEVELOPMENT@AIRBNB.PUBLIC>put file:///home/davar/Downloads/zips2000.csv @AIRBNB.PUBLIC.%zipcodes2000_snowsql;
zips2000.csv_c.gz(0.33MB): [##########] 100.00% Done (0.474s, 0.70MB/s).        
+--------------+-----------------+-------------+-------------+--------------------+--------------------+----------+---------+
| source       | target          | source_size | target_size | source_compression | target_compression | status   | message |
|--------------+-----------------+-------------+-------------+--------------------+--------------------+----------+---------|
| zips2000.csv | zips2000.csv.gz |      987372 |      346309 | NONE               | GZIP               | UPLOADED |         |
+--------------+-----------------+-------------+-------------+--------------------+--------------------+----------+---------+
1 Row(s) produced. Time Elapsed: 2.006s
ADAVARSKI#DEVELOPMENT@AIRBNB.PUBLIC>copy into zipcodes2000_snowsql from @%zipcodes2000_snowsql file_format = (type = csv field_optionally_enclosed_by='"' SKIP_HEADER = 1);
+-----------------+--------+-------------+-------------+-------------+-------------+-------------+------------------+-----------------------+-------------------------+
| file            | status | rows_parsed | rows_loaded | error_limit | errors_seen | first_error | first_error_line | first_error_character | first_error_column_name |
|-----------------+--------+-------------+-------------+-------------+-------------+-------------+------------------+-----------------------+-------------------------|
| zips2000.csv.gz | LOADED |       42192 |       42192 |           1 |           0 | NULL        |             NULL |                  NULL | NULL                    |
+-----------------+--------+-------------+-------------+-------------+-------------+-------------+------------------+-----------------------+-------------------------+
1 Row(s) produced. Time Elapsed: 3.644s
ADAVARSKI#DEVELOPMENT@AIRBNB.PUBLIC>
ADAVARSKI#DEVELOPMENT@AIRBNB.PUBLIC>select count(*) from ZIPCODES2000_SNOWSQL;
+----------+                                                                    
| COUNT(*) |
|----------|
|    42192 |
+----------+
1 Row(s) produced. Time Elapsed: 0.605s
ADAVARSKI#DEVELOPMENT@AIRBNB.PUBLIC>

```
Check Snowflake UI:

DWH:

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Snowflake-UI-DW-DEVELOPMENT.png" width="800">

DB:AIRBNB:TABLE:ZIPCODES2000_SNOWSQL

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Snowflake-Databrick-write-sampletable.png" width="800">


Continuous Data Loading with Snowpipe

```
#create a new database for testing snowpipe
create database snowpipe data_retention_time_in_days = 1;
show databases like 'snow%';
# create a new external stage
create or replace stage snowpipe.public.snowstage
url='S3://<your_s3_bucket>'
credentials=(
AWS_KEY_ID='<your_AWS_KEY_ID>',
AWS_SECRET_KEY='<your_AWS_SEKRET_KEY>');
# create target table for Snowpipe
create or replace table snowpipe.public.snowtable(
    jsontext variant
);
# create a new pipe
create or replace pipe snowpipe.public.snowpipe
    auto_ingest=true as
            copy into snowpipe.public.snowtable
            from @snowpipe.public.snowstage
            file_format = (type = 'JSON');
#Note:Variant is universal semistructured data type of Snowflake
for loading data in formats such as JSON, Avro, ORC, Parquet, or
XML. 

# check exists pipes and stages
show pipes;
show stages;
# Copy the SQS ARN link from the NotificationChannel field.
# Using a simple select statement, we can check the count of
loaded data.
# check count of rows in target table
select count(*) from snowpipe.public.snowtable
```


Data flow process between Snowflake data warehouse services and managed Apache Spark/Databricks.

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Snowflake-Spark-bidirectional-data-transfer.png" width="800">

Snowflake <-> Spark interaction:

<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Snowfalke-spark-interaction-table.png" width="600">


Databricks (notebook):

Create a new notebook using Databricks ➤ Create a blank notebook, call it snowflake_airbnb, and attach the existing
cluster(MLOps).

Connect to Snowflake.

Replace the substitutions according to your Snowflake credentials
before executing

```
options = dict(sfUrl="vw60186.eu-central-1.snowflakecomputing.com",
   sfUser="ADAVARSKI",
   sfPassword="Kr0k0dil!",
   sfDatabase= "AIRBNB",
   sfSchema= "PUBLIC",
   sfWarehouse= "DEVELOPMENT" )
```
Read data from Snowflake.
```

df = spark.read \
  .format("snowflake") \
  .options(**options) \
  .option("dbtable", "ZIPCODES2000_SNOWSQL") \
  .load()
display(df)
```
Write data into Snowflake.
```
df.write \
    .format("snowflake") \
    .options(**options) \
    .option("dbtable", "sampletable") \
    .save() \
```

Databricks:

<img src="https://github.com/adavarski/SaaS-ML-k8s/blob/main/k8s/Demo6-Spark-ML/pictures/Snowflake-Databricks-simple-notebook.png" width="800">

Notebook example:

https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/snowflake/snowflake_airbnb.ipynb

Snowflake:

<img src="https://github.com/adavarski/SaaS-ML-k8s/blob/main/k8s/Demo6-Spark-ML/pictures/Snowflake-Databrick-write-sampletable.png" width="800">

Modern Analytics Solution/Platform architecture example with Snowflake DWH:


<img src="https://github.com/adavarski/Databricks-Snowflake-ML-playground/blob/main/pictures/Snowflake-modern-DW-architecture.png" width="800">

Note: Snowflake helped us to leverage big data and streaming capabilities that were impossible with the legacy solution. For big data, we were processing web logs for example within Apache Spark deployed on top of the EMR cluster. Snowflake accesses Parquet files, and we don’t need to load them into Snowflake. For the streaming use case, we leveraged DynamoDB streams and Kinesis Firehose, and all data is sent into an S3 bucket where Snowflake can consume it.

Ref: https://github.com/adavarski/AWS-UAP (WIP)
