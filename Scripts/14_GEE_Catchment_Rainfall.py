"""
================================================================================
14_GEE_Catchment_Rainfall.py
Workflow Step 14: Upstream Catchment Rainfall Extraction over HydroBASINS Level 12
Manuscript Section: 2.5.2 Vegetation and rainfall data

Description:
  Extracts 36-year (1990-2025) annual wet-season and 90-day antecedent rainfall
  for every HydroBASINS Level-12 sub-catchment across the Ugab River basin in
  Google Earth Engine:
    - Primary Wet-Season Rainfall (November 1 - April 30)
    - 90-Day Antecedent Rainfall (May 1 - July 31)

  In Step 15, these polygon-level rainfall totals are area-weighted across the
  entire upstream contributing drainage area for each evaluation reach.

Inputs:
  - 'Data/Ugab_Downstream_Reaches_514.geojson' (from Step 12, to define basin bounds)
  - HydroBASINS Level 12 FeatureCollection ('WWF/HydroSHEDS/v1/Basins/hybas_12')
  - CHIRPS DAILY Precipitation Collection ('UCSB-CHG/CHIRPS/DAILY')

Outputs:
  - Exports to Google Drive: 'Ugab_HYBAS12_Catchment_Rainfall_1990_2025.csv'
================================================================================
"""

import os
import json
import ee

# ============================================================
# 0. CONFIGURATION & INITIALISATION
# ============================================================
# Set your local 'Data' folder path (e.g., r"C:\path\to\Data")
DATA_DIR = r"YOUR_PATH_TO_DATA_FOLDER"
os.makedirs(DATA_DIR, exist_ok=True)

# Replace 'YOUR_GEE_PROJECT' with your Google Earth Engine project ID
GEE_PROJECT = 'YOUR_GEE_PROJECT'

try:
    ee.Initialize(project=GEE_PROJECT)
    print("Earth Engine initialised successfully.")
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
    print("Earth Engine authenticated and initialised successfully.")

# ============================================================
# 1. LOAD BASIN BOUNDS FROM REACH GEOMETRY
# ============================================================
geojson_path = os.path.join(DATA_DIR, 'Ugab_Downstream_Reaches_514.geojson')

print(f"Loading reach geometries to determine catchment bounds from: {geojson_path}")
with open(geojson_path, 'r') as f:
    fc_json = json.load(f)

def strip_z(coords):
    if not coords:
        return coords
    if isinstance(coords[0], (int, float)):
        return coords[:2]
    return [strip_z(c) for c in coords]

features = []
for feat in fc_json['features']:
    geom_dict = feat['geometry']
    if 'coordinates' in geom_dict:
        geom_dict['coordinates'] = strip_z(geom_dict['coordinates'])
    features.append(ee.Feature(ee.Geometry(geom_dict), feat.get('properties', {})))

fc_zones = ee.FeatureCollection(features)
basin_bounds = fc_zones.geometry().bounds()

# Filter HydroBASINS Level 12 sub-catchments covering the Ugab basin
basins = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_12").filterBounds(basin_bounds)
print("Filtered HydroBASINS Level 12 sub-catchments.")

# ============================================================
# 2. CHIRPS RAINFALL COMPOSITING (1990-2025)
# ============================================================
chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
years = ee.List.sequence(1990, 2025)

def process_year_rain(year):
    y = ee.Number(year)
    
    # Wet-season (Nov 1 prev year to Apr 30 current year)
    wet_start = ee.Date.fromYMD(y.subtract(1), 11, 1)
    wet_end = ee.Date.fromYMD(y, 4, 30)
    rain_wet = chirps.filterDate(wet_start, wet_end).sum().rename('Rain_WetSeason_Catchment')
    
    # 90-day antecedent to dry season (May 1 to Jul 31)
    ant_start = ee.Date.fromYMD(y, 5, 1)
    ant_end = ee.Date.fromYMD(y, 7, 31)
    rain_ant = chirps.filterDate(ant_start, ant_end).sum().rename('Rain_Antecedent90_Catchment')
    
    return ee.Image([rain_wet, rain_ant]).set('Year', y)

rain_col = ee.ImageCollection(years.map(process_year_rain))

# ============================================================
# 3. REDUCE OVER HYDROBASINS LEVEL 12 POLYGONS & EXPORT
# ============================================================
print("Setting up Earth Engine reduction across HydroBASINS Level 12 polygons...")

def extract_rain(img):
    y = img.get('Year')
    stats = img.reduceRegions(
        collection=basins,
        reducer=ee.Reducer.mean(),
        scale=5000,
        tileScale=4
    )
    return stats.map(lambda f: f.set('Year', y))

final_rain = rain_col.map(extract_rain).flatten()

export_task = ee.batch.Export.table.toDrive(
    collection=final_rain,
    description='Ugab_HYBAS12_Catchment_Rainfall_1990_2025',
    fileNamePrefix='Ugab_HYBAS12_Catchment_Rainfall_1990_2025',
    folder='EarthEngineExports',
    fileFormat='CSV',
    selectors=['HYBAS_ID', 'SUB_AREA', 'Year', 'Rain_WetSeason_Catchment', 'Rain_Antecedent90_Catchment']
)

export_task.start()
print(f"Catchment Rainfall Export Task submitted! Task ID: {export_task.id}")
print("Monitor status in GEE Tasks tab. Once complete, download 'Ugab_HYBAS12_Catchment_Rainfall_1990_2025.csv' into the 'Data' folder to run Step 15.")
