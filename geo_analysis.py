# geo_analysis.py
"""
Simplified Geo-Spatial Analytics: Heatmap & Hotspot Detection (using pandas)
This version processes GPS data without requiring PySpark.
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import json

# Config
INPUT_CSV = "data/gps_traces.csv"
OUTPUT_DIR = "output"
GRID_DELTA_LAT = 0.005
GRID_DELTA_LON = 0.005
TOP_N_HOTSPOTS = 20
KMEANS_K = 6
DBSCAN_EPS = 0.002
DBSCAN_MIN_SAMPLES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("Loading GPS data...")
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=['latitude', 'longitude'])
    print(f"Total points: {len(df)}")
    
    # Grid-based heatmap
    print("Creating grid-based heatmap...")
    df['cell_x'] = (df['latitude'] / GRID_DELTA_LAT).apply(np.floor).astype(int)
    df['cell_y'] = (df['longitude'] / GRID_DELTA_LON).apply(np.floor).astype(int)
    
    # Aggregate by grid cell
    grid_agg = df.groupby(['cell_x', 'cell_y']).agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'point_id': 'count'
    }).reset_index()
    
    grid_agg.columns = ['cell_x', 'cell_y', 'center_lat', 'center_lon', 'count']
    grid_agg['grid_id'] = grid_agg['cell_x'].astype(str) + '_' + grid_agg['cell_y'].astype(str)
    grid_agg = grid_agg.sort_values('count', ascending=False)
    
    # Save grid counts
    grid_out_path = os.path.join(OUTPUT_DIR, "grid_counts.csv")
    grid_agg.to_csv(grid_out_path, index=False)
    print(f"Saved grid counts to: {grid_out_path}")
    
    # Save top hotspot grids
    top_grids = grid_agg.head(TOP_N_HOTSPOTS)
    top_hotspots_csv = os.path.join(OUTPUT_DIR, "top_hotspot_grids.csv")
    top_grids[['grid_id', 'center_lat', 'center_lon', 'count']].to_csv(top_hotspots_csv, index=False)
    print(f"Saved top hotspot grids to: {top_hotspots_csv}")
    
    # KMeans clustering
    print("Running KMeans clustering...")
    coords = df[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=KMEANS_K, random_state=42, n_init=10)
    df['kmeans_cluster'] = kmeans.fit_predict(coords)
    
    # Save KMeans centroids
    kmeans_out = os.path.join(OUTPUT_DIR, "kmeans_centroids.csv")
    centroids_df = pd.DataFrame({
        'cluster': range(KMEANS_K),
        'centroid_lat': kmeans.cluster_centers_[:, 0],
        'centroid_lon': kmeans.cluster_centers_[:, 1]
    })
    centroids_df.to_csv(kmeans_out, index=False)
    print(f"Saved KMeans centroids to: {kmeans_out}")
    
    # Save cluster counts
    cluster_counts = df.groupby('kmeans_cluster').size().reset_index(name='count')
    cluster_counts.columns = ['prediction', 'count']
    clustered_out = os.path.join(OUTPUT_DIR, "kmeans_cluster_counts.csv")
    cluster_counts.to_csv(clustered_out, index=False)
    print(f"Saved cluster counts to: {clustered_out}")
    
    # DBSCAN clustering
    print("Running DBSCAN clustering...")
    # Sample points for DBSCAN (to speed up processing)
    sample_size = min(3000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    sample_coords = sample_df[['latitude', 'longitude']].values
    
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='euclidean')
    labels = dbscan.fit_predict(sample_coords)
    sample_df['dbscan_cluster'] = labels
    
    # Find DBSCAN hotspots (excluding noise label -1)
    clusters = sample_df[sample_df['dbscan_cluster'] >= 0].groupby('dbscan_cluster').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'point_id': 'count'
    }).reset_index()
    clusters.columns = ['cluster', 'latitude', 'longitude', 'count']
    clusters = clusters.sort_values('count', ascending=False)
    
    dbscan_out = os.path.join(OUTPUT_DIR, "dbscan_hotspots.csv")
    clusters.to_csv(dbscan_out, index=False)
    print(f"Saved DBSCAN hotspots to: {dbscan_out}")
    
    # Create JSON for web visualization
    print("Creating JSON files for web visualization...")
    
    # All GPS points (sample for performance)
    sample_all = df.sample(n=min(5000, len(df)), random_state=42)
    gps_data = sample_all[['latitude', 'longitude', 'vehicle_id']].to_dict('records')
    with open(os.path.join(OUTPUT_DIR, 'gps_points.json'), 'w') as f:
        json.dump(gps_data, f)
    
    # Grid heatmap data
    grid_data = grid_agg[['center_lat', 'center_lon', 'count']].to_dict('records')
    with open(os.path.join(OUTPUT_DIR, 'grid_heatmap.json'), 'w') as f:
        json.dump(grid_data, f)
    
    # KMeans centroids
    kmeans_data = centroids_df.to_dict('records')
    with open(os.path.join(OUTPUT_DIR, 'kmeans_centroids.json'), 'w') as f:
        json.dump(kmeans_data, f)
    
    # DBSCAN hotspots
    dbscan_data = clusters.to_dict('records')
    with open(os.path.join(OUTPUT_DIR, 'dbscan_hotspots.json'), 'w') as f:
        json.dump(dbscan_data, f)
    
    print("Analysis complete!")
    print(f"- Total GPS points: {len(df)}")
    print(f"- Grid cells: {len(grid_agg)}")
    print(f"- KMeans clusters: {KMEANS_K}")
    print(f"- DBSCAN clusters found: {len(clusters)}")

if __name__ == "__main__":
    main()
