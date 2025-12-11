# generate_gps_data.py
# Generates synthetic GPS traces (CSV) for multiple vehicles clustered around hotspots.

import csv
import random
from datetime import datetime, timedelta

OUT_FILE = "data/gps_traces.csv"
NUM_POINTS = 20000          # total GPS points
NUM_VEHICLES = 200
HOTSPOT_CENTERS = [
    (12.9716, 77.5946),     # center A (e.g., downtown)
    (12.9352, 77.6245),     # center B
    (13.0156, 77.5970),     # center C
]
HOTSPOT_RADIUS = 0.01      # approx ~1 km radius (in degrees)
NON_HOTSPOT_SPREAD = 0.05  # wide area noise

random.seed(42)

def jitter(center_lat, center_lon, radius):
    return center_lat + random.uniform(-radius, radius), center_lon + random.uniform(-radius, radius)

def main():
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_id", "vehicle_id", "latitude", "longitude", "timestamp"])
        base_time = datetime.utcnow()
        for i in range(NUM_POINTS):
            # choose whether this point is in a hotspot (70% chance) or noise
            if random.random() < 0.7:
                center = random.choice(HOTSPOT_CENTERS)
                lat, lon = jitter(center[0], center[1], HOTSPOT_RADIUS)
            else:
                # random around a larger bounding box near Bangalore region
                lat = 12.85 + random.random() * 0.35   # ~12.85 to 13.20
                lon = 77.50 + random.random() * 0.25   # ~77.50 to 77.75

            vehicle_id = "veh_{:04d}".format(random.randint(1, NUM_VEHICLES))
            ts = (base_time - timedelta(seconds=random.randint(0, 3600))).isoformat()
            writer.writerow([i, vehicle_id, round(lat, 6), round(lon, 6), ts])

    print(f"Generated {NUM_POINTS} GPS points -> {OUT_FILE}")

if __name__ == "__main__":
    main()
