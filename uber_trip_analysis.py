# 📦 Import Libraries
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Ensure Spark is available and configured ---
if "JAVA_HOME" not in os.environ:
    os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17.0.13"

if not os.path.exists(os.environ["JAVA_HOME"]):
    raise EnvironmentError(
        f"JAVA_HOME is set to '{os.environ['JAVA_HOME']}', but this directory does not exist. "
        "Please check your Java installation and JAVA_HOME environment variable."
    )

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import (
        col, to_timestamp, hour, dayofmonth, month, dayofweek, count, expr,
        coalesce, trim, to_date, unix_timestamp, lit
    )
except Exception as e:
    print("PySpark import failed. Make sure PySpark is installed and JAVA_HOME is set correctly.")
    raise e

# 📂 Start Spark Session
try:
    spark = SparkSession.builder.appName("UberTripAnalysis").getOrCreate()
except Exception as e:
    print("Failed to start SparkSession. Check your Java installation and JAVA_HOME.")
    raise e

# 📂 Load Data using Spark
data_paths = [
    r"D:\ch_notebook\notebooks\uber-raw-data-apr14.csv\uber-raw-data-apr14.csv",
    r"D:\ch_notebook\notebooks\uber-raw-data-may14.csv\uber-raw-data-may14.csv",
    r"D:\ch_notebook\notebooks\uber-raw-data-jun14.csv\uber-raw-data-jun14.csv",
    r"D:\ch_notebook\notebooks\uber-raw-data-jul14.csv\uber-raw-data-jul14.csv",
    r"D:\ch_notebook\notebooks\uber-raw-data-aug14.csv\uber-raw-data-aug14.csv",
    r"D:\ch_notebook\notebooks\uber-raw-data-sep14.csv\uber-raw-data-sep14.csv"
]

try:
    # Read with explicit schema to ensure proper types
    df = spark.read.csv(data_paths, header=True, inferSchema=False)
    # Force all columns to be read as strings initially
    df = df.select(
        col("Date/Time").cast("string").alias("Date/Time"),
        col("Lat").cast("string").alias("Lat"),
        col("Lon").cast("string").alias("Lon"),
        col("Base").cast("string").alias("Base")
    )
except Exception as e:
    print("Failed to read CSV files with Spark. Check file paths and permissions.")
    raise e

# 🧹 Clean & Prepare with Spark
df = df.na.drop(subset=["Date/Time", "Lat", "Lon"])
df = df.withColumn("Date/Time", trim(col("Date/Time")))

# Parse Date/Time using a more robust approach
# First try common formats, then fall back to simpler parsing
df = df.withColumn(
    "DateTime",
    coalesce(
        to_timestamp(col("Date/Time"), "M/d/yyyy H:mm:ss"),
        to_timestamp(col("Date/Time"), "M/d/yyyy HH:mm:ss"),
        to_timestamp(col("Date/Time"), "MM/dd/yyyy HH:mm:ss"),
        # Fallback - parse as Unix timestamp if all else fails
        to_timestamp(col("Date/Time"))
    )
)

# Filter out any rows where DateTime couldn't be parsed
df = df.filter(col("DateTime").isNotNull())

# Convert Lat/Lon to numeric types
df = df.withColumn("Lat", col("Lat").cast("double"))
df = df.withColumn("Lon", col("Lon").cast("double"))

# Add time features using Spark
df = df.withColumn("Hour", hour(col("DateTime")))
df = df.withColumn("Day", dayofmonth(col("DateTime")))
df = df.withColumn("Weekday", dayofweek(col("DateTime")))  # 1=Sunday, 7=Saturday
df = df.withColumn("Month", month(col("DateTime")))

# ✅ Show Sample from Spark DataFrame
print("Sample of the data (from Spark DataFrame):")
df.select("DateTime", "Hour", "Weekday", "Month", "Lat", "Lon").show(5, truncate=False)

# --- 1. Distribution Figure: Distribution of trips per hour (using Spark) ---
hourly_counts = df.groupBy("Hour").agg(count("*").alias("NumTrips")).orderBy("Hour")
hourly_pd = hourly_counts.toPandas()

plt.figure(figsize=(10,6))
sns.barplot(x="Hour", y="NumTrips", data=hourly_pd, color='skyblue')
plt.title('Distribution of Uber Trips by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Trips')
plt.xticks(range(0,24))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- 2. Outliers Figure: Boxplot of Latitude and Longitude (using Spark sample) ---
latlon_pd = df.select("Lat", "Lon").sample(fraction=0.01, seed=42).toPandas()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.boxplot(y=latlon_pd['Lat'], color='lightcoral')
plt.title('Boxplot of Latitude (Outlier Detection)')
plt.ylabel('Latitude')
plt.subplot(1,2,2)
sns.boxplot(y=latlon_pd['Lon'], color='lightgreen')
plt.title('Boxplot of Longitude (Outlier Detection)')
plt.ylabel('Longitude')
plt.tight_layout()
plt.show()

# --- 3. Relationships Figure: Pairplot of time and location features (using Spark sample) ---
sample_pd = df.select("Hour", "Day", "Month", "Lat", "Lon").sample(fraction=0.01, seed=42).toPandas()
sns.pairplot(sample_pd, diag_kind='kde', plot_kws={'alpha':0.3})
plt.suptitle('Relationships between Time and Location Features', y=1.02)
plt.show()

# Stop Spark session when done
spark.stop()