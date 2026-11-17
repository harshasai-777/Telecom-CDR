"""
spark_pipeline.py
=================
Distributed Telecom CDR Analysis Pipeline on Apache Spark.

Demonstrates the enterprise-scale counterpart to the Dask local pipeline:
- Apache Spark (PySpark) DataFrame for distributed heterogeneous CDR loading
- Spark MLlib VectorAssembler & StandardScaler
- Distributed KMeans clustering on Spark
- Distributed XGBoost Classifier via `xgboost.spark.SparkXGBClassifier`

This script can be submitted to any Spark cluster (Local, YARN, Kubernetes, or Databricks):
    spark-submit --packages ml.dmlc:xgboost4j-spark_2.12:1.7.0 spark_pipeline.py
"""

import sys
import os

def run_spark_pipeline(data_path: str = "data/cdr_small.csv"):
    """
    Execute end-to-end distributed pipeline using PySpark and XGBoost on Spark.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.ml.feature import VectorAssembler, StandardScaler
        from pyspark.ml.clustering import KMeans
        from pyspark.ml.evaluation import ClusteringEvaluator, MulticlassClassificationEvaluator
        from xgboost.spark import SparkXGBClassifier
    except ImportError:
        print("[Spark Pipeline Info]")
        print("PySpark or xgboost.spark is not installed in the local single-machine Python environment.")
        print("This module provides the enterprise multi-node cluster architecture implementation.")
        print("To run on a real Spark cluster:")
        print("  1. Install PySpark: pip install pyspark")
        print("  2. Submit: spark-submit spark_pipeline.py")
        return

    # 1. Initialize Spark Session with Distributed Cluster Configuration
    print("1. Initializing Distributed SparkSession ...")
    spark = SparkSession.builder \
        .appName("Telecom-CDR-Distributed-Analytics") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    print(f"2. Reading Heterogeneous CDR Big Data from {data_path} ...")
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # 2. Distributed Feature Engineering using Spark SQL functions
    print("3. Distributed Feature Engineering on Spark DataFrame ...")
    df = df.withColumn("call_start_ts", F.to_timestamp("call_start")) \
           .withColumn("hour", F.hour("call_start_ts")) \
           .withColumn("day_of_week", F.dayofweek("call_start_ts")) \
           .withColumn("duration_min", F.round(df["duration_sec"] / 60.0, 2)) \
           .withColumn("is_night", F.when((F.col("hour") >= 22) | (F.col("hour") < 6), 1).otherwise(0))

    # 3. Subscriber Aggregations at Big Data Scale
    print("4. Aggregating Features per Subscriber (Distributed GroupBy) ...")
    sub_df = df.groupBy("caller_id").agg(
        F.count("record_id").alias("total_calls"),
        F.sum("duration_min").alias("total_duration_min"),
        F.mean("duration_min").alias("avg_duration_min"),
        F.mean("is_night").alias("pct_night_calls"),
        F.mean("is_international").alias("pct_international"),
        F.mean("is_roaming").alias("pct_roaming"),
        F.countDistinct("cell_tower_id").alias("unique_towers"),
        F.countDistinct("callee_id").alias("unique_callees"),
        F.sum("charge_usd").alias("total_charge_usd"),
        F.max("is_fraud").alias("is_fraud")
    ).na.fill(0)

    feature_cols = [
        "total_calls", "total_duration_min", "avg_duration_min",
        "pct_night_calls", "pct_international", "pct_roaming",
        "unique_towers", "unique_callees", "total_charge_usd"
    ]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    sub_df_vec = assembler.transform(sub_df)

    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    scaler_model = scaler.fit(sub_df_vec)
    sub_scaled = scaler_model.transform(sub_df_vec)

    # 4. Distributed K-Means Clustering on Spark
    print("5. Distributed K-Means Clustering via Spark MLlib ...")
    kmeans = KMeans(k=5, seed=42, featuresCol="features", predictionCol="cluster")
    kmeans_model = kmeans.fit(sub_scaled)
    clustered_df = kmeans_model.transform(sub_scaled)

    # 5. Distributed XGBoost Classifier (xgboost.spark)
    print("6. Distributed XGBoost Classifier Training on Spark ...")
    train_df, test_df = clustered_df.randomSplit([0.8, 0.2], seed=42)

    xgb = SparkXGBClassifier(
        features_col="features",
        label_col="is_fraud",
        num_workers=2,
        max_depth=6,
        n_estimators=100,
        learning_rate=0.1
    )
    xgb_model = xgb.fit(train_df)
    predictions = xgb_model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(labelCol="is_fraud", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Distributed Spark XGBoost Accuracy: {accuracy:.4f}")

    spark.stop()
    print("Distributed Spark pipeline execution finished successfully.")

if __name__ == "__main__":
    run_spark_pipeline()
