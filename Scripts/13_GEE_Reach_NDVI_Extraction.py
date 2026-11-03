"""
================================================================================
13_GEE_Reach_NDVI_Extraction.py
Workflow Step 13: Multi-Decadal Landsat Vegetation & Water Metrics in Earth Engine
Manuscript Section: 2.5.2 Vegetation and rainfall data

Description:
  Extracts 36-year (1990-2025) annual vegetation, water, and climate indicators
  across all 514 river reaches in Google Earth Engine:
    - Late dry-season NDVI & NDMI (August 1 - October 31): Captures riparian
      vegetation condition during peak water stress.
    - Post-wet season NDVI (March 1 - May 31): Captures green flush after floods.
    - Annual Maximum and Annual Mean (AUC proxy) NDVI.
    - MNDWI Water Inundation Frequency (MNDWI > 0): Flags submerged reaches.
    - Local antecedent wet-season rainfall (Nov 1 - Apr 30) from CHIRPS DAILY.

Inputs:
  - 'Data/Ugab_Downstream_Reaches_514.geojson' (from Step 12)

Outputs:
  - Exports to Google Drive: 'Ugab_Reach_Vegetation_Metrics_1990_2025.csv'
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
# 1. LOAD EVALUATION REACH GEOMETRIES
# ============================================================
geojson_path = os.path.join(DATA_DIR, 'Ugab_Downstream_Reaches_514.geojson')

print(f"Loading reach geometries from: {geojson_path}")
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

fc = ee.FeatureCollection(features)
print(f"Constructed Earth Engine FeatureCollection with {len(features)} reaches.")

# ============================================================
# 2. LANDSAT HARMONIZATION (LANDSAT 5, 7, 8, 9)
# ============================================================
def rename_l57(img):
    return img.select(
        ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'],
        ['Blue',  'Green', 'Red',   'NIR',   'SWIR1', 'SWIR2', 'QA_PIXEL']
    )

def rename_l89(img):
    return img.select(
        ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'],
        ['Blue',  'Green', 'Red',   'NIR',   'SWIR1', 'SWIR2', 'QA_PIXEL']
    )

def apply_scale_factors(img):
    optical = img.select(['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']).multiply(0.0000275).add(-0.2)
    return img.addBands(optical, None, True)

def mask_clouds(img):
    qa = img.select('QA_PIXEL')
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return img.updateMask(cloud_mask)

def add_indices(img):
    ndvi = img.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    ndmi = img.normalizedDifference(['NIR', 'SWIR1']).rename('NDMI')
    mndwi = img.normalizedDifference(['Green', 'SWIR1']).rename('MNDWI')
    return img.addBands([ndvi, ndmi, mndwi])

def get_landsat(start_date, end_date):
    l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterDate(start_date, end_date).map(rename_l57)
    l7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterDate(start_date, end_date).map(rename_l57)
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(start_date, end_date).map(rename_l89)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterDate(start_date, end_date).map(rename_l89)
    merged = l5.merge(l7).merge(l8).merge(l9)
    return merged.map(apply_scale_factors).map(mask_clouds).map(add_indices)

# ============================================================
# 3. ANNUAL TIME-SERIES COMPOSITING (1990-2025)
# ============================================================
years = ee.List.sequence(1990, 2025)
chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")

def process_year(y):
    year = ee.Number(y)
    
    # 1. Rainfall: Wet season (Nov 1 prev year to Apr 30) & 90-day antecedent to dry (May 1 to Jul 31)
    wet_start = ee.Date.fromYMD(year.subtract(1), 11, 1)
    wet_end = ee.Date.fromYMD(year, 4, 30)
    rain_wet = chirps.filterDate(wet_start, wet_end).sum().rename('Rain_WetSeason')
    
    ant_start = ee.Date.fromYMD(year, 5, 1)
    ant_end = ee.Date.fromYMD(year, 7, 31)
    rain_ant = chirps.filterDate(ant_start, ant_end).sum().rename('Rain_Antecedent90')
    
    # 2. Dry season (Aug 1 - Oct 31)
    dry_start = ee.Date.fromYMD(year, 8, 1)
    dry_end = ee.Date.fromYMD(year, 10, 31)
    dry_col = get_landsat(dry_start, dry_end)
    ndvi_dry = dry_col.select('NDVI').median().rename('NDVI_Dry')
    ndmi_dry = dry_col.select('NDMI').median().rename('NDMI_Dry')
    
    # 3. Post-wet season flush (Mar 1 - May 31)
    pw_start = ee.Date.fromYMD(year, 3, 1)
    pw_end = ee.Date.fromYMD(year, 5, 31)
    pw_col = get_landsat(pw_start, pw_end)
    ndvi_pw = pw_col.select('NDVI').median().rename('NDVI_PostWet')
    
    # 4. Annual Maximum and Mean NDVI
    ann_start = ee.Date.fromYMD(year, 1, 1)
    ann_end = ee.Date.fromYMD(year, 12, 31)
    ann_col = get_landsat(ann_start, ann_end)
    ndvi_max = ann_col.select('NDVI').reduce(ee.Reducer.percentile([95])).rename('NDVI_Max')
    ndvi_mean = ann_col.select('NDVI').mean().rename('NDVI_Mean')
    
    # 5. MNDWI Water Inundation Frequency (MNDWI > 0)
    mndwi = ann_col.select('MNDWI')
    water_freq = mndwi.map(lambda img: img.gt(0)).mean().rename('MNDWI_Freq')
    
    combined = ee.Image([
        rain_wet, rain_ant,
        ndvi_dry, ndmi_dry,
        ndvi_pw, ndvi_max, ndvi_mean,
        water_freq
    ]).set('year', year)
    
    return combined

yearly_col = ee.ImageCollection(years.map(process_year))

# ============================================================
# 4. REDUCE ACROSS REACHES & EXPORT TO GOOGLE DRIVE
# ============================================================
print("Setting up Earth Engine reduction across reaches...")

def extract_regions(img):
    y = img.get('year')
    stats = img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=30,
        tileScale=4
    )
    return stats.map(lambda f: f.set('Year', y))

final_fc = yearly_col.map(extract_regions).flatten()

export_task = ee.batch.Export.table.toDrive(
    collection=final_fc,
    description='Ugab_Reach_Vegetation_Metrics_1990_2025',
    fileNamePrefix='Ugab_Reach_Vegetation_Metrics_1990_2025',
    folder='EarthEngineExports',
    fileFormat='CSV',
    selectors=['zone_id', 'type', 'dam_id', 'HYRIV_ID', 'Year',
               'Rain_WetSeason', 'Rain_Antecedent90',
               'NDVI_Dry', 'NDMI_Dry', 'NDVI_PostWet',
               'NDVI_Max', 'NDVI_Mean', 'MNDWI_Freq']
)

export_task.start()
print(f"Export task submitted! Task ID: {export_task.id}")
print("Monitor status in GEE Tasks tab. Once complete, download 'Ugab_Reach_Vegetation_Metrics_1990_2025.csv' to the 'Data' folder to run Step 15.")
