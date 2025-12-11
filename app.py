# app.py
"""
Flask web application for Geo-Spatial Analytics Visualization
"""

from flask import Flask, render_template, jsonify, send_from_directory
import json
import os

app = Flask(__name__)

# Paths
OUTPUT_DIR = "output"

@app.route('/')
def index():
    """Main page with interactive map"""
    return render_template('index.html')

@app.route('/api/gps_points')
def get_gps_points():
    """Get GPS points data"""
    try:
        with open(os.path.join(OUTPUT_DIR, 'gps_points.json'), 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/grid_heatmap')
def get_grid_heatmap():
    """Get grid heatmap data"""
    try:
        with open(os.path.join(OUTPUT_DIR, 'grid_heatmap.json'), 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/kmeans_centroids')
def get_kmeans_centroids():
    """Get KMeans cluster centroids"""
    try:
        with open(os.path.join(OUTPUT_DIR, 'kmeans_centroids.json'), 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dbscan_hotspots')
def get_dbscan_hotspots():
    """Get DBSCAN hotspot clusters"""
    try:
        with open(os.path.join(OUTPUT_DIR, 'dbscan_hotspots.json'), 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get analysis statistics"""
    try:
        stats = {}
        
        # GPS points count
        with open(os.path.join(OUTPUT_DIR, 'gps_points.json'), 'r') as f:
            gps_data = json.load(f)
            stats['gps_points'] = len(gps_data)
        
        # Grid cells count
        with open(os.path.join(OUTPUT_DIR, 'grid_heatmap.json'), 'r') as f:
            grid_data = json.load(f)
            stats['grid_cells'] = len(grid_data)
        
        # KMeans clusters
        with open(os.path.join(OUTPUT_DIR, 'kmeans_centroids.json'), 'r') as f:
            kmeans_data = json.load(f)
            stats['kmeans_clusters'] = len(kmeans_data)
        
        # DBSCAN clusters
        with open(os.path.join(OUTPUT_DIR, 'dbscan_hotspots.json'), 'r') as f:
            dbscan_data = json.load(f)
            stats['dbscan_clusters'] = len(dbscan_data)
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
