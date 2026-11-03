"""
================================================================================
01_GEE_PointPixel_Extraction.py
Workflow Step 01: Extract Point-Pixel Training Features in Google Earth Engine
Manuscript Section: 2.3.1 Data extraction and detection metrics

Description:
  Extracts 21 Sentinel-2 optical, Sentinel-1 SAR backscatter, and topographical
  (SRTM DEM slope/elevation, MERIT Hydro HAND) features across candidate pixels
  in the Swakop River basin. Samples include:
    1. Active reference dams (Atlas of Namibia validated, n = 1,159)
    2. False positive hard negatives across iterative testing rounds (n = 1,300)
    3. Stratified background landcover samples from ESA WorldCover v200 (n = 2,500)

Output:
  Exports CSV to Google Drive: 'Swakop_PointPixel_Training_Data.csv'
================================================================================
"""

import os
import ee
import geopandas as gpd
import shapely

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

# Local Vector Training Assets
swakop_basin_path = os.path.join(DATA_DIR, 'Swakop_Basin_Boundary.gpkg')
atlas_path = os.path.join(DATA_DIR, 'Swakop_Reference_Dams.gpkg')
hard_negatives_path = os.path.join(DATA_DIR, 'Swakop_Hard_Negatives.gpkg')

# ============================================================
# 1. PREPARE TRAINING GEOMETRIES (DAMS, HARD NEGATIVES, BACKGROUND)
# ============================================================
print("Loading training vector geometries from Data folder...")
swakop_gdf = gpd.read_file(swakop_basin_path).to_crs("EPSG:4326")
swakop_gdf.geometry = shapely.force_2d(swakop_gdf.geometry)
swakop_roi = ee.Geometry(swakop_gdf.geometry.iloc[0].__geo_interface__)

# A. Active Reference Dams (Class = 1)
dams_gdf = gpd.read_file(atlas_path).to_crs("EPSG:4326")
dams_gdf.geometry = shapely.force_2d(dams_gdf.geometry)
dam_features = [ee.Feature(ee.Geometry.Point([geom.x, geom.y])) for geom in dams_gdf.geometry]
dams_fc = ee.FeatureCollection(dam_features).map(lambda f: f.set({'Class': 1}))

# B. Hard Negatives from Iterative Testing Rounds (Class = 0)
hn_gdf = gpd.read_file(hard_negatives_path).to_crs("EPSG:4326")
hn_pts = shapely.force_2d(hn_gdf.geometry.centroid)
hn_features = [ee.Feature(ee.Geometry.Point([geom.x, geom.y])) for geom in hn_pts]
hn_fc = ee.FeatureCollection(hn_features).map(lambda f: f.set({'Class': 0}))

# C. Stratified Background Samples from ESA WorldCover (Class = 0)
exclusion_zone = dams_fc.map(lambda f: f.buffer(100))
exclusion_mask = ee.Image.constant(1).paint(exclusion_zone, 0)

worldcover = ee.ImageCollection("ESA/WorldCover/v200").mosaic().clip(swakop_roi)
valid_classes = worldcover.neq(50).And(worldcover.neq(80))  # Exclude urban (50) and permanent water (80)
background_mask = valid_classes.updateMask(exclusion_mask)

# Stratified sampling across trees (10), shrubland (20), grassland (30), cropland (40), bare (60)
background_points = worldcover.updateMask(background_mask).stratifiedSample(
    numPoints=0,
    classBand='Map',
    region=swakop_roi,
    scale=100,
    classValues=[10, 20, 30, 40, 60],
    classPoints=[50, 600, 600, 250, 1000],
    geometries=True
).map(lambda f: f.set({'Class': 0}))

training_geoms = dams_fc.merge(hn_fc).merge(background_points)
print("Master training geometries assembled (Reference Dams + Hard Negatives + Stratified Background).")

# ============================================================
# 2. SENTINEL-2 OPTICAL FEATURE EXTRACTION
# ============================================================
print("Processing Sentinel-2 optical composites...")
s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(swakop_roi) \
    .filterDate('2016-01-01', '2025-01-01')

cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
s2_masked = s2.linkCollection(cs_plus, ['cs_cdf']).map(
    lambda img: img.updateMask(img.select('cs_cdf').gte(0.60))
)

# Late wet season (Jan-Mar) & early dry season (May-Jul)
s2_wet = s2_masked.filter(ee.Filter.calendarRange(1, 3, 'month'))
s2_dry = s2_masked.filter(ee.Filter.calendarRange(5, 7, 'month'))

def get_s2_metrics(img):
    mndwi = img.normalizedDifference(['B3', 'B11']).rename('MNDWI')
    ndmi = img.normalizedDifference(['B8', 'B11']).rename('NDMI')
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    mndwi_freq = mndwi.gt(0.1).rename('MNDWI_freq')
    ndmi_freq = ndmi.gt(0.0).rename('NDMI_freq')
    return img.addBands([mndwi, ndmi, ndvi, mndwi_freq, ndmi_freq])

s2_wet_mapped = s2_wet.map(get_s2_metrics)
s2_dry_mapped = s2_dry.map(get_s2_metrics)

# Wet Season Metrics
optical_wet_median = s2_wet_mapped.select(['MNDWI', 'NDMI', 'NDVI']).median().rename(['MNDWI_wet_median', 'NDMI_wet_median', 'NDVI_wet_median'])
optical_wet_freq = s2_wet_mapped.select(['MNDWI_freq', 'NDMI_freq']).mean().rename(['MNDWI_wet_freq', 'NDMI_wet_freq'])
spatial_water_fraction_wet = optical_wet_median.select('MNDWI_wet_median').gt(0.1).rename('spatial_water_fraction_wet')
n_s2_wet = s2_wet_mapped.select('MNDWI').count().rename('n_s2_wet')

# Dry Season Metrics
optical_dry_median = s2_dry_mapped.select(['MNDWI', 'NDMI', 'NDVI']).median().rename(['MNDWI_dry_median', 'NDMI_dry_median', 'NDVI_dry_median'])
optical_dry_freq = s2_dry_mapped.select(['MNDWI_freq', 'NDMI_freq']).mean().rename(['MNDWI_dry_freq', 'NDMI_dry_freq'])
spatial_water_fraction_dry = optical_dry_median.select('MNDWI_dry_median').gt(0.1).rename('spatial_water_fraction_dry')
n_s2_dry = s2_dry_mapped.select('MNDWI').count().rename('n_s2_dry')

# ============================================================
# 3. SENTINEL-1 SAR FEATURE EXTRACTION
# ============================================================
print("Processing Sentinel-1 SAR composites...")
s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
    .filterBounds(swakop_roi) \
    .filterDate('2016-01-01', '2025-01-01') \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
    .filter(ee.Filter.eq('instrumentMode', 'IW'))

def calc_ratio(img):
    vv_minus_vh = img.select('VV').subtract(img.select('VH')).rename('VV_minus_VH')
    return img.addBands(vv_minus_vh)

s1_features = s1.map(calc_ratio)
s1_wet = s1_features.filter(ee.Filter.calendarRange(1, 3, 'month'))
s1_dry = s1_features.filter(ee.Filter.calendarRange(5, 7, 'month'))

sar_wet_median = s1_wet.select(['VV', 'VH', 'VV_minus_VH']).median().rename(['VV_wet_median', 'VH_wet_median', 'VVVH_wet_median'])
sar_dry_median = s1_dry.select(['VV', 'VH', 'VV_minus_VH']).median().rename(['VV_dry_median', 'VH_dry_median', 'VVVH_dry_median'])
n_s1_wet = s1_wet.select('VV').count().rename('n_s1_wet')
n_s1_dry = s1_dry.select('VV').count().rename('n_s1_dry')

# ============================================================
# 4. TOPOGRAPHICAL FEATURES
# ============================================================
srtm = ee.Image('USGS/SRTMGL1_003')
slope = ee.Terrain.slope(srtm).rename('slope')
elevation = srtm.rename('elv')
merit_hydro = ee.Image("MERIT/Hydro/v1_0_1")
hand = merit_hydro.select('hnd').rename('hand')

# ============================================================
# 5. COMBINE AND EXTRACT MASTER TRAINING DATA
# ============================================================
feature_stack = ee.Image.cat([
    optical_wet_median, optical_wet_freq, spatial_water_fraction_wet, n_s2_wet,
    optical_dry_median, optical_dry_freq, spatial_water_fraction_dry, n_s2_dry,
    sar_wet_median, n_s1_wet,
    sar_dry_median, n_s1_dry,
    slope, elevation, hand
])

print("Extracting feature values across training points...")
extracted_data = feature_stack.reduceRegions(
    collection=training_geoms,
    reducer=ee.Reducer.first(),
    scale=10,
    tileScale=4
)

export_task = ee.batch.Export.table.toDrive(
    collection=extracted_data,
    description='Swakop_PointPixel_Training_Data',
    fileNamePrefix='Swakop_PointPixel_Training_Data',
    folder='EarthEngineExports',
    fileFormat='CSV'
)

export_task.start()
print(f"Export task submitted! Task ID: {export_task.id}")
print("Monitor status in the Google Earth Engine Tasks tab. Once complete, download 'Swakop_PointPixel_Training_Data.csv' to your 'Data' folder to run Step 02.")
