# geo_heatmap_hotspots.py
"""
Geo-Spatial Analytics: Heatmap & Hotspot Detection (PySpark)

What it does:
 - Loads CSV of GPS points (point_id, vehicle_id, latitude, longitude, timestamp)
 - Produces a grid-based heatmap by aggregating points into latitude/longitude bins
 - Saves grid counts as CSV (grid_id, count, center_lat, center_lon)
 - Detects hotspots using:
     * Spark MLlib KMeans (scalable)
     * Optional DBSCAN on sampled points (density-based) -- runs on driver
 - Saves cluster centroids and top-N hotspot grid cells to CSV
 - Optionally plots heatmap (collects grid counts to driver and uses matplotlib)

Usage:
    spark-submit geo_heatmap_hotspots.py
or (local):
    python geo_heatmap_hotspots.py

Dependencies:
    - pyspark
    - numpy, pandas, scikit-learn (for DBSCAN and plotting; DBSCAN optional)
    - matplotlib (optional, for plotting)
"""

import os
from math import floor
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, floor as spark_floor, concat_ws, lit, avg, count, expr
from pyspark.sql.types import DoubleType
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

# Optional libraries for plotting / DBSCAN (install in driver env)
try:
    import pandas as pd
    import numpy as np
    from sklearn.cluster import DBSCAN
    import matplotlib.pyplot as plt
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------------- Config ----------------
INPUT_CSV = "data/gps_traces.csv"         # produced by generate_gps_data.py
OUTPUT_DIR = "output"
GRID_DELTA_LAT = 0.005    # degrees (≈ 0.005 ~ 0.55 km lat)
GRID_DELTA_LON = 0.005    # degrees
TOP_N_HOTSPOTS = 20
KMEANS_K = 6              # number of clusters for KMeans hotspot detection
SAMPLE_FOR_DBSCAN = 3000  # number of points to sample for DBSCAN (runs on driver)
DBSCAN_EPS = 0.002        # approx degrees -> tune for density (0.002 ~ 200m)
DBSCAN_MIN_SAMPLES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Spark session ----------------
spark = SparkSession.builder \
    .appName("GeoHeatmapHotspots") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ---------------- Load data ----------------
df = spark.read.csv(INPUT_CSV, header=True, inferSchema=True) \
     .select("point_id", "vehicle_id", col("latitude").cast(DoubleType()), col("longitude").cast(DoubleType()), "timestamp")

df = df.na.drop(subset=["latitude", "longitude"])
print("Total points:", df.count())

# ---------------- Grid aggregation (heatmap) ----------------
# Compute grid cell indices for each point
# We compute integer cell indices using fixed delta degrees.
df_with_grid = df.withColumn("cell_x", spark_floor((col("latitude") / GRID_DELTA_LAT)).cast("long")) \
                 .withColumn("cell_y", spark_floor((col("longitude") / GRID_DELTA_LON)).cast("long"))

# Aggregate counts per grid cell; compute center lat/lon
grid_agg = df_with_grid.groupBy("cell_x", "cell_y") \
            .agg(
                count("*").alias("count"),
                avg("latitude").alias("center_lat"),
                avg("longitude").alias("center_lon")
            ) \
            .orderBy(col("count").desc())

# Add a grid_id string
grid_agg = grid_agg.withColumn("grid_id", concat_ws("_", col("cell_x"), col("cell_y"))) \
                   .select("grid_id", "cell_x", "cell_y", "center_lat", "center_lon", "count")

# Save grid counts to CSV
grid_out_path = os.path.join(OUTPUT_DIR, "grid_counts.csv")
grid_agg.coalesce(1).write.mode("overwrite").option("header", "true").csv(grid_out_path)
print("Saved grid counts to:", grid_out_path)

# Also save top-N hotspots by grid count
top_grids = grid_agg.limit(TOP_N_HOTSPOTS).collect()
top_hotspots_csv = os.path.join(OUTPUT_DIR, "top_hotspot_grids.csv")
with open(top_hotspots_csv, "w") as f:
    f.write("grid_id,center_lat,center_lon,count\n")
    for r in top_grids:
        f.write(f"{r['grid_id']},{r['center_lat']},{r['center_lon']},{r['count']}\n")
print("Saved top hotspot grids to:", top_hotspots_csv)

# ---------------- KMeans clustering for hotspots (scalable) ----------------
# Prepare features
assembler = VectorAssembler(inputCols=["latitude", "longitude"], outputCol="features")
points_with_features = assembler.transform(df.select("latitude", "longitude").na.drop())

kmeans = KMeans(k=KMEANS_K, seed=42, featuresCol="features", predictionCol="cluster")
kmeans_model = kmeans.fit(points_with_features)
centers = kmeans_model.clusterCenters()

# Save KMeans centroids
kmeans_out = os.path.join(OUTPUT_DIR, "kmeans_centroids.csv")
with open(kmeans_out, "w") as f:
    f.write("cluster,centroid_lat,centroid_lon\n")
    for i, c in enumerate(centers):
        # c is [lat, lon]
        f.write(f"{i},{float(c[0])},{float(c[1])}\n")
print("Saved KMeans centroids to:", kmeans_out)

# Save cluster counts (how many points per cluster)
clustered = kmeans_model.transform(points_with_features).groupBy("prediction").count().orderBy("prediction")
clustered_out = os.path.join(OUTPUT_DIR, "kmeans_cluster_counts.csv")
clustered.coalesce(1).write.mode("overwrite").option("header", "true").csv(clustered_out)
print("Saved cluster counts to:", clustered_out)

# ---------------- Optional: DBSCAN for density-based hotspots (on sampled points) ----------------
if SKLEARN_AVAILABLE:
    # sample points to driver (for DBSCAN)
    total_points = df.count()
    sample_frac = min(1.0, SAMPLE_FOR_DBSCAN / max(1, total_points))
    sample = df.sample(withReplacement=False, fraction=sample_frac, seed=42).select("latitude", "longitude").toPandas()

    coords = sample[["latitude", "longitude"]].values
    print(f"Running DBSCAN on {len(coords)} sampled points (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})")
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="haversine" if False else "euclidean").fit(coords)
    labels = db.labels_
    sample["cluster"] = labels

    # find clusters (ignoring noise label -1)
    clusters = sample[sample["cluster"] >= 0].groupby("cluster").agg({
        "latitude":"mean",
        "longitude":"mean",
        "cluster":"count"
    }).rename(columns={"cluster":"count"}).reset_index().sort_values("count", ascending=False)

    dbscan_out = os.path.join(OUTPUT_DIR, "dbscan_hotspots.csv")
    clusters.to_csv(dbscan_out, index=False)
    print("Saved DBSCAN hotspots to:", dbscan_out)
else:
    print("scikit-learn/pandas not available: skipping DBSCAN density clustering (optional).")

# ---------------- Optional: Quick plot of heatmap (driver-side) ----------------
try:
    # collect grid counts to pandas and plot a scatter heatmap with marker size = count
    grid_pd = grid_agg.select("center_lon", "center_lat", "count").toPandas()
    if not grid_pd.empty:
        plt.figure(figsize=(8,8))
        plt.scatter(grid_pd["center_lon"], grid_pd["center_lat"], s=grid_pd["count"], alpha=0.6)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Grid-based Heatmap (marker size ~ count)")
        heatmap_png = os.path.join(OUTPUT_DIR, "heatmap_scatter.png")
        plt.savefig(heatmap_png, dpi=150, bbox_inches="tight")
        print("Saved heatmap plot to:", heatmap_png)
    else:
        print("Grid is empty — no plot produced.")
except Exception as e:
    print("Plotting failed (matplotlib/pandas may be unavailable):", e)

# ---------------- Finish ----------------
spark.stop()
print("Done.")
